# WM9B7 PowerGraph — Pre-build & GitHub release instructions

This walks the §5.4 workflow end-to-end: from a clean Azure compute target
to a public GitHub release whose URL and SHA-256 you can paste into the
notebook's cell 7. Plan time: **~1.5 hours total**, of which ~40 minutes is
unattended training on Azure.

---

## What you'll produce

| Output | Where | Size |
|---|---|---|
| `ieee24_processed.pt` | Azure → local | ~40 MB |
| `ieee24_gine_best.pt` | Azure → local | ~100 KB |
| `ieee24_split_indices.json` | Azure → local | ~250 KB |
| `submission_bundle.tar.gz` | local → GitHub Release | ~50 MB |

The notebook reads the bundle from a public GitHub Release URL. Once the
release is live you only need to update two strings in cell 7:

```python
BUNDLE_URL = "https://github.com/<USERNAME>/wm9b7-powergraph-submission/releases/download/v1.0/submission_bundle.tar.gz"
EXPECTED_SHA256 = "<paste the SHA the assembler prints>"
```

---

## Files in this kit

| File | Where it runs | What it does |
|---|---|---|
| `azure_prebuild.py` | Azure VM (T4) | Downloads the PowerGraph dataset, processes it, trains GINe, persists the split. Produces the three artefacts above. |
| `assemble_bundle.py` | Local machine | Combines those three artefacts with your local results JSONs and figure PNGs into a deterministic `.tar.gz`, prints the SHA-256. |
| `requirements_prebuild.txt` | Azure VM | Two extra packages that AzureML's stock kernel doesn't ship. |
| `PREBUILD_INSTRUCTIONS.md` | (this file) | The runbook. |

You will also need, on the Azure VM, the **untrimmed** versions of:

- `powergraph_data.py` (the full one with raw `.mat` handling — about 437 lines)
- `powergraph_models.py` (the full one with `GINe_Graph` plus the other
  architectures — about 329 lines; only `GINe_Graph` is used at training time
  but the full file imports cleanly without trouble)
- `train_ieee24_baselines.py`

These already exist in your project repo. **Don't use the trimmed
notebook-submission versions on Azure** — those have the raw-`.mat` path
removed and will fail at phase 2.

---

## Phase A — Azure pre-build (~50 minutes)

### A1. Spin up the compute target

In Azure ML Studio:

1. **Compute → Create → Compute instance**.
2. **Virtual machine size:** `Standard_NC6s_v3` (Tesla T4 GPU, 16 GB VRAM).
   This matches the notebook's marker-time target so any
   compute-architecture surprises surface here, not at marking.
3. **Image:** `Python 3.10 - AzureML` (the same kernel the notebook uses).
4. **Disk:** the default 120 GB OS disk is fine. We need ~10 GB peak (the
   2.7 GB tarball + ~700 MB extracted IEEE-24 raw + ~40 MB processed +
   training intermediates).
5. Once it's running, open a **terminal** (not a notebook) on the instance.

### A2. Stage the working directory

```bash
mkdir -p ~/wm9b7-prebuild && cd ~/wm9b7-prebuild
```

Upload, into this folder:

| File | Source |
|---|---|
| `azure_prebuild.py` | this kit |
| `requirements_prebuild.txt` | this kit |
| `powergraph_data.py` | **full** version from your project repo |
| `powergraph_models.py` | **full** version from your project repo |
| `train_ieee24_baselines.py` | from your project repo |

(Use the AzureML file-browser, `scp`, or `git clone` your project repo and
copy the four files in.)

### A3. Install dependencies

The PyG ecosystem needs to match the resident torch + CUDA. The
`Python 3.10 - AzureML` kernel ships a recent torch with CUDA, so derive
the wheel index from it rather than pinning a version that may be wrong:

```bash
python - <<'PY'
import torch
v = torch.__version__.split("+")[0]
cu = ("cu" + torch.version.cuda.replace(".", "")) if torch.cuda.is_available() else "cpu"
print(f"https://data.pyg.org/whl/torch-{v}+{cu}.html")
PY
# Copy the URL it prints, then:
pip install -q torch-scatter torch-sparse -f <THE_URL_FROM_ABOVE>
pip install -q -r requirements_prebuild.txt
```

Smoke check:

```bash
python -c "import torch, torch_geometric, torch_scatter, torch_sparse, h5py, scipy; print('OK', torch.__version__, torch.cuda.is_available())"
```

If it prints `OK <version> True` you're done with setup.

### A4. Run the pre-build

```bash
python azure_prebuild.py 2>&1 | tee prebuild.log
```

Expect five phases; full log will be ~150 lines. Approximate timings on a
T4:

| Phase | What it does | ~Time |
|---|---|---|
| 1 | Download Figshare tarball, extract IEEE-24 raw `.mat` files | 3–5 min |
| 2 | `PowerGraphDataset(...)` → `ieee24_binary.pt` | 3–5 min |
| 3 | Train GINe seed=23 with Optuna-best HPs (200 epochs, patience 30; usually stops near epoch 120) | 25–40 min |
| 4 | Write split JSON with keys `"train"`, `"val"`, `"test"` | <1 s |
| 5 | Verify all three outputs (load checkpoint into a fresh model, forward pass) | <10 s |

Re-running is safe and **fast**: each phase skips if its output already
exists. A re-run that only rewrites the split JSON takes seconds. Useful
flags if something fails partway:

- `--skip-download` — assume raw `.mat` files are already in place.
- `--skip-train` — skip training (e.g. if you only want to regenerate the
  split JSON).
- `--force-train` — re-train even if the checkpoint already exists.

When phase 5 prints `[OK ]` for all three outputs, the pre-build is done.

### A5. Download the artefacts

The three outputs sit in `~/wm9b7-prebuild/prebuild_outputs/`. Pull them
back to your local machine — easiest is the AzureML file-browser, or:

```bash
# from your local machine
scp -r azureuser@<your-vm-ip>:~/wm9b7-prebuild/prebuild_outputs ./
```

You can shut down the Azure compute instance at this point. Phases B–E
all run locally.

---

## Phase B — Assemble the bundle locally (~30 seconds)

You'll combine the three Azure outputs with the JSONs and figures already
sitting in your project's `artifacts/` tree.

### B1. Layout

The assembler's most flexible mode takes one umbrella `--artifacts-dir`
and recursively finds each named JSON / PNG inside it. If your project
keeps everything under one top-level `artifacts/` directory, that's all
you need:

```bash
python assemble_bundle.py \
    --azure-out      ./prebuild_outputs \
    --artifacts-dir  ./artifacts \
    --output         ./submission_bundle.tar.gz
```

If your JSONs and figures live in different roots, pass them explicitly:

```bash
python assemble_bundle.py \
    --azure-out   ./prebuild_outputs \
    --results-dir ./artifacts \
    --figures-dir ./artifacts \
    --output      ./submission_bundle.tar.gz
```

### B2. Resolve & verify the manifest first

Before writing the tarball, dry-run the resolver to catch missing files
early:

```bash
python assemble_bundle.py \
    --azure-out      ./prebuild_outputs \
    --artifacts-dir  ./artifacts \
    --print-manifest-only
```

Expected: 56 entries resolved (3 root files + 23 JSONs + 30 PNGs), zero
missing. If anything's missing, the assembler prints `MISSING: <name>` for
each and refuses to build. Re-run with the right paths or generate the
missing artefact, then try again.

### B3. Write the bundle

Drop `--print-manifest-only` and run for real. The assembler prints:

```
Bundle:    ./submission_bundle.tar.gz
Size:      ~50,000,000 B (~48 MB)
SHA-256:   <64 hex chars>

Paste these into the notebook (cell 7, the BUNDLE_URL / EXPECTED_SHA256
placeholders):

    BUNDLE_URL = "https://github.com/<USERNAME>/wm9b7-powergraph-submission/releases/download/v1.0/submission_bundle.tar.gz"
    EXPECTED_SHA256 = "<the same 64 hex chars>"
```

**Determinism note:** the same inputs produce a byte-identical tarball on
any machine, so the SHA above is reproducible. If you ever rebuild and
the SHA changes, an input file changed (or the manifest did) — investigate
before re-uploading.

---

## Phase C — Create the GitHub release (~5 minutes)

### C1. Repo

If you haven't yet:

```bash
# with the GitHub CLI
gh auth login            # one-time
gh repo create wm9b7-powergraph-submission --public --description \
    "WM9B7 PowerGraph submission bundle (Varbella et al., NeurIPS 2024, CC BY 4.0)"
```

Or via the web UI at https://github.com/new (set visibility to **Public**;
private won't work because the notebook downloads anonymously).

Add a one-line `README.md` so the repo isn't bare. The CC BY 4.0 licence
on PowerGraph requires attribution — drop it in the README:

```bash
cd wm9b7-powergraph-submission
cat > README.md <<'EOF'
# WM9B7 PowerGraph submission bundle

Hosts the precomputed artefact bundle consumed by `WM9B7_PowerGraph.ipynb`.

The IEEE-24 cascade-failure data inside the bundle is derived from the
**PowerGraph** benchmark (Varbella, Briola, Aste, Cremer & Mountanios,
NeurIPS 2024), distributed under **CC BY 4.0**:
<https://doi.org/10.6084/m9.figshare.22820534>.

The bundle additionally contains a GINe checkpoint and result summaries
produced by this submission's training and analysis scripts.
EOF
git add README.md && git commit -m "Initial README with PowerGraph attribution"
git push -u origin main
```

### C2. Release with the bundle as an asset

```bash
gh release create v1.0 \
    submission_bundle.tar.gz \
    --title "v1.0 — WM9B7 submission bundle" \
    --notes "Bundle for WM9B7_PowerGraph.ipynb. SHA-256: <paste from assembler>"
```

Or via the web UI: **Releases → Draft new release**, tag `v1.0`, attach
`submission_bundle.tar.gz` to the release, click **Publish release**.

The asset URL is:

```
https://github.com/<USERNAME>/wm9b7-powergraph-submission/releases/download/v1.0/submission_bundle.tar.gz
```

(GitHub's URL pattern is contractual — `download/<tag>/<filename>` — so
you can paste this into the notebook before the upload finishes if you
prefer.)

### C3. Sanity-check the public URL

From any machine that's not logged into GitHub:

```bash
curl -sLI "<the URL>" | grep -iE "^(HTTP|content-length|content-type)"
```

Expected: a `200 OK` (after one or two redirects) and a content-length
near 50 MB. If you see `404` the asset isn't attached; if you see a login
page the repo isn't public.

---

## Phase D — Update the notebook (~1 minute)

In `WM9B7_PowerGraph.ipynb`, cell 7, replace the placeholders:

```python
BUNDLE_URL = "https://github.com/<USERNAME>/wm9b7-powergraph-submission/releases/download/v1.0/submission_bundle.tar.gz"
EXPECTED_SHA256 = "<the SHA-256 the assembler printed>"
```

Save the notebook.

---

## Phase E — Azure dry-run (~5 minutes wall, includes ~1 min cold start)

This is the marker-equivalent test. Same compute target as A1 (T4,
`Python 3.10 - AzureML`):

1. Upload `WM9B7_PowerGraph.ipynb`, `powergraph_models.py` (trimmed
   submission version), `powergraph_data.py` (trimmed submission version)
   to a fresh folder on the Azure instance.
2. Open the notebook with kernel `Python 3.10 - AzureML`.
3. **Run All.**

Confirm, in order:

| Check | Where to look | Expected |
|---|---|---|
| (a) PyG wheel install succeeds | cell 2 output | no errors; `torch-scatter`, `torch-sparse`, `torch-geometric==2.6.1`, `captum` install quietly |
| (b) bundle SHA matches | cell 4 output | no `AssertionError`; `Bundle ready. 23 JSONs, 30 figures.` |
| (c) cell 17 finishes inside ~2 min | cell 17 wall-clock | watch the live training cell complete with epoch logs |
| (d) checkpoint loads on T4 | cell 10 / load_state_dict | no `RuntimeError`, no missing/unexpected keys; eval metrics print |

If (a) fails: the AzureML stock torch may have rolled to a version PyG
hasn't pre-built wheels for. Fall back to source build with `pip install
torch-scatter torch-sparse --no-binary=:all:` (slow, ~10 min) or pin a
slightly older PyG to use a known wheel.

If (b) fails: the bundle on disk doesn't match `EXPECTED_SHA256`. Either
the upload was corrupted (re-upload), or you regenerated the bundle and
forgot to update the SHA in the notebook.

If (c) is slow: the live training in cell 9 (3 epochs) takes ~90 s on T4.
If a cell takes much longer, check that CUDA is actually picked up
(`torch.cuda.is_available()` in cell 2 should print True).

If (d) fails: this would be the surprise the user listed in the original
plan. Standard PyTorch state_dicts are device-portable, so a clean
`load_state_dict` failure usually means the model architecture in the
notebook doesn't match the one used to train the checkpoint. Recheck the
cell 8 hyperparameters against `OPTUNA_BEST_GINE` in `azure_prebuild.py`
— `hidden_dim=64`, `num_layers=4`, `pooling="mean_max"`, `dropout≈0.126`.

When all four checks pass, the submission is ready.

---

## Troubleshooting cribsheet

**`tarfile.ReadError: not a gzip file`** during phase 1 of the prebuild.
The Figshare download was interrupted. Delete `powergraph_data.tar.gz`
and re-run; the script will re-download.

**`KeyError: ...` during PowerGraphDataset.process()** in phase 2. One of
the IEEE-24 `.mat` files is from a different grid (e.g. `of_bi.mat` from
IEEE-39 ended up in `dataset_cascades/ieee24/ieee24/raw/`). Wipe
`dataset_cascades/` and re-run from phase 1.

**Training collapses to BalAcc ≈ 0.5** in phase 3. The class-weighted
focal loss is sensitive to seed; if you re-trained with a different seed
and got a degenerate run, set `seed=23` explicitly (the default) — that's
the seed the rest of the project's results were obtained with, and the
one the live notebook trains with.

**`split JSON missing required key 'train'`** during phase 5 verify. You
edited the JSON by hand or the script's split writer was modified.
Re-run phase 4 (`python azure_prebuild.py --skip-download --skip-train`)
to regenerate.

**Assembler reports `MISSING: results/<name>.json`** locally. The named
JSON wasn't generated yet. Either run the experiment that produces it
(scripts in your project repo), or — if the JSON genuinely doesn't apply
to this submission — remove its name from `RESULTS_JSONS` in
`assemble_bundle.py` (and the corresponding cell in the notebook).

**SHA changes between rebuilds with no input change.** Means a manifest
entry's bytes changed silently. Check `git status` on the artifacts
directory; a regenerated figure or JSON often has new bytes even when
"same content" is true (matplotlib timestamps, JSON key ordering). The
deterministic tarring is doing its job — what differed was upstream.
