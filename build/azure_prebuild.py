"""WM9B7 PowerGraph — Azure pre-build script.

Generates the three artefacts that the submission bundle ships out of the
notebook:

    - ieee24_processed.pt        (PyG InMemoryDataset, IEEE-24 binary task)
    - ieee24_gine_best.pt        (GINe checkpoint, seed=23, Optuna-best HPs,
                                  200 epochs with early stopping)
    - ieee24_split_indices.json  (deterministic 85/5/10 split on seed=23,
                                  with the keys "train", "val", "test" that
                                  the notebook hard-codes in cell 7)

Implements §5.4 steps 1-4 of the build plan. Steps 5-7 (download, local
bundle assembly, SHA-256) live in `assemble_bundle.py`.

Usage on Azure (Standard NC6s_v3, T4):

    python azure_prebuild.py

By default the script runs every phase end-to-end. Each phase is idempotent
— re-running it skips work whose output already exists. Pass --skip-train
or --skip-download for partial reruns.

Expected wall-clock on a T4: download ~3-5 min, .mat → .pt conversion
~3-5 min, training ~25-40 min (200 epochs with patience=30, typical
convergence ≈ epoch 120). Total ≈ 35-50 minutes.

Dependencies (the AzureML `Python 3.10 - AzureML` kernel ships most of these;
the rest are listed in `requirements_prebuild.txt`):
    torch, torch-geometric, torch-scatter, torch-sparse, scipy, numpy,
    scikit-learn, h5py, pandas
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# PowerGraph (Varbella et al., NeurIPS 2024) — graph-level cascade dataset.
# DOI: 10.6084/m9.figshare.22820534, CC BY 4.0.
FIGSHARE_DOWNLOAD_URL = "https://figshare.com/ndownloader/files/46619158"
FIGSHARE_TARBALL_NAME = "powergraph_data.tar.gz"

# These are the seven raw matrices PowerGraphDataset.process() consumes.
RAW_FILES = ["blist.mat", "Bf.mat", "Ef.mat",
             "of_bi.mat", "of_mc.mat", "of_reg.mat", "exp.mat"]

# Where PowerGraphDataset (full module) expects raw files, mirroring the
# rest of the project's scripts (e.g. eda_ieee24.py, train_ieee24_baselines.py).
DATASET_ROOT = Path("dataset_cascades/ieee24/ieee24")
RAW_DIR = DATASET_ROOT / "raw"
PROCESSED_PT_SRC = DATASET_ROOT / "processed" / "ieee24_binary.pt"

# Training output (set by train_ieee24_baselines.run_training).
TRAINED_CKPT_SRC = Path("artifacts/training/ieee24_gine/best_model.pt")

# Final outputs that get downloaded back to the submission machine.
OUT_DIR = Path("prebuild_outputs")
OUT_DATASET = OUT_DIR / "ieee24_processed.pt"
OUT_CHECKPOINT = OUT_DIR / "ieee24_gine_best.pt"
OUT_SPLIT_JSON = OUT_DIR / "ieee24_split_indices.json"

# Optuna-best GINe hyperparameters, copied verbatim from
# run_architecture_comparison.OPTUNA_BEST_GINE so the checkpoint matches
# the architecture_comparison_summary.json results the notebook cites.
OPTUNA_BEST_GINE = {
    "architecture": "GINe",
    "hidden_dim": 64,
    "num_layers": 4,
    "pooling": "mean_max",
    "lr": 0.0015304852121831463,
    "weight_decay": 1.3783237455007196e-06,
    "focal_gamma": 2.018862129753596,
    "alpha_cap": 7.557861855309373,
    "dropout": 0.12602063719411183,
    "epochs": 200,
    "patience": 30,
    "batch_size": 16,
    "heads": 4,
    "grid": "ieee24",
    "task": "binary",
    "seed": 23,
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}", flush=True)


def step(msg: str) -> None:
    print(f"  [+] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"  [!] {msg}", flush=True)


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# --------------------------------------------------------------------------
# Phase 1 — Raw data
# --------------------------------------------------------------------------

def have_raw_files() -> bool:
    return all((RAW_DIR / f).exists() for f in RAW_FILES)


def download_figshare(dest: Path) -> None:
    """Download the PowerGraph tarball from Figshare. Prefer wget for speed
    on multi-GB downloads, fall back to urllib if wget is missing."""
    if dest.exists() and dest.stat().st_size > 0:
        step(f"Tarball already cached at {dest} "
             f"({fmt_bytes(dest.stat().st_size)}). Skipping download.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    step(f"Downloading PowerGraph dataset (~2.7 GB) → {dest}")
    t0 = time.time()

    if shutil.which("wget"):
        subprocess.run(
            ["wget", "--quiet", "--show-progress", "--tries=3",
             "-O", str(dest), FIGSHARE_DOWNLOAD_URL],
            check=True,
        )
    else:
        warn("wget not found; falling back to urllib (slower, no progress bar).")
        urllib.request.urlretrieve(FIGSHARE_DOWNLOAD_URL, str(dest))

    step(f"Download finished in {time.time() - t0:.1f}s "
         f"({fmt_bytes(dest.stat().st_size)}).")


def extract_ieee24_raw(tarball: Path, target_raw_dir: Path) -> None:
    """Extract just the IEEE-24 .mat files we need. The Figshare archive
    contains all four grids (~3.78 GB uncompressed); we only need ~700 MB
    for IEEE-24, so this saves disk and time."""
    target_raw_dir.mkdir(parents=True, exist_ok=True)

    step(f"Inspecting tarball at {tarball}")
    with tarfile.open(tarball, "r:gz") as tf:
        members = tf.getmembers()

        # Find the IEEE-24 subdirectory in the archive. The archive layout is
        # not contractually fixed — different Figshare uploads have shipped
        # both `cascades/ieee24/raw/<file>.mat` and
        # `dataset_cascades/ieee24/ieee24/raw/<file>.mat`. Locate by basename.
        wanted_basenames = set(RAW_FILES)
        ieee24_members: dict[str, tarfile.TarInfo] = {}
        for m in members:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            if base not in wanted_basenames:
                continue
            # We want the IEEE-24 copy specifically, not (e.g.) the same
            # filename for IEEE-39 / UK / IEEE-118.
            if "ieee24" not in m.name.lower():
                continue
            # First match wins (avoid overwriting if archive nests duplicates).
            ieee24_members.setdefault(base, m)

        missing = wanted_basenames - set(ieee24_members)
        if missing:
            raise RuntimeError(
                f"Could not find these IEEE-24 raw files inside the tarball: "
                f"{sorted(missing)}.\n"
                f"Listing of .mat members in the archive:\n  " +
                "\n  ".join(m.name for m in members
                            if m.isfile() and m.name.endswith('.mat'))
            )

        for base, m in sorted(ieee24_members.items()):
            out_path = target_raw_dir / base
            step(f"Extracting {m.name} → {out_path} "
                 f"({fmt_bytes(m.size)})")
            with tf.extractfile(m) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)


# --------------------------------------------------------------------------
# Phase 2 — Process raw .mat into PyG .pt
# --------------------------------------------------------------------------

def build_processed_dataset():
    """Instantiate PowerGraphDataset once; PyG's InMemoryDataset machinery
    will run `process()` and write `<root>/processed/ieee24_binary.pt`.
    Returns the dataset object so phase 4 can compute splits without
    reloading from disk."""
    # Imported here so the script can still print --help without PyG.
    from powergraph_data import PowerGraphDataset  # type: ignore

    step("Constructing PowerGraphDataset(grid='ieee24', task='binary'). "
         "First call processes raw .mat → .pt (~3-5 min on T4).")
    t0 = time.time()
    dataset = PowerGraphDataset(
        root=str(DATASET_ROOT),
        grid_name="ieee24",
        task="binary",
    )
    step(f"Dataset ready in {time.time() - t0:.1f}s. "
         f"|D|={len(dataset)} graphs.")
    return dataset


def copy_processed_pt() -> None:
    if not PROCESSED_PT_SRC.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_PT_SRC}. "
            "PowerGraphDataset.__init__ should have produced this."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROCESSED_PT_SRC, OUT_DATASET)
    step(f"Copied {PROCESSED_PT_SRC} → {OUT_DATASET} "
         f"({fmt_bytes(OUT_DATASET.stat().st_size)}).")


# --------------------------------------------------------------------------
# Phase 3 — Train GINe (seed=23, Optuna-best, 200 epochs)
# --------------------------------------------------------------------------

def train_gine(skip_if_exists: bool = True) -> None:
    if skip_if_exists and OUT_CHECKPOINT.exists():
        step(f"Checkpoint already at {OUT_CHECKPOINT} "
             f"({fmt_bytes(OUT_CHECKPOINT.stat().st_size)}). "
             "Skipping training (pass --force-train to override).")
        return

    # Imported here so the script can still print --help without these.
    from train_ieee24_baselines import TrainConfig, run_training  # type: ignore

    cfg = TrainConfig(**OPTUNA_BEST_GINE)
    step(f"Training GINe with Optuna-best HPs:")
    for k, v in OPTUNA_BEST_GINE.items():
        print(f"        {k}={v}")

    t0 = time.time()
    # enable_mlops=False keeps the prebuild self-contained (no MLflow tracking
    # server needed on the Azure box).
    result = run_training(cfg, enable_mlops=False)
    elapsed = time.time() - t0

    step(f"Training complete in {elapsed/60:.1f} min. "
         f"best_val_bal_acc={result['best_val_bal_acc']:.4f} "
         f"@ epoch {result['best_epoch']}")
    step(f"Test BalAcc={result['test_metrics']['bal_acc']:.4f}, "
         f"PR-AUC={result['test_metrics']['pr_auc']:.4f}.")

    if not TRAINED_CKPT_SRC.exists():
        raise FileNotFoundError(
            f"Expected checkpoint at {TRAINED_CKPT_SRC} but it was not created."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRAINED_CKPT_SRC, OUT_CHECKPOINT)
    step(f"Copied {TRAINED_CKPT_SRC} → {OUT_CHECKPOINT} "
         f"({fmt_bytes(OUT_CHECKPOINT.stat().st_size)}).")


# --------------------------------------------------------------------------
# Phase 4 — Persist deterministic split
# --------------------------------------------------------------------------

def write_split_json(dataset=None) -> None:
    """Persist 85/5/10 stratified split for seed=23 with the JSON keys the
    notebook hard-codes (`train`, `val`, `test`)."""
    from powergraph_data import (  # type: ignore
        PowerGraphDataset, SplitConfig, split_dataset_indices,
    )

    if dataset is None:
        # Re-load (fast — InMemoryDataset just reads the .pt back).
        dataset = PowerGraphDataset(
            root=str(DATASET_ROOT),
            grid_name="ieee24",
            task="binary",
        )

    cfg = SplitConfig(random_state=23)
    train_idx, val_idx, test_idx = split_dataset_indices(dataset, cfg)

    payload = {
        # ---- The keys the notebook reads. Don't rename. -------------------
        "train": list(map(int, train_idx)),
        "val": list(map(int, val_idx)),
        "test": list(map(int, test_idx)),
        # ---- Metadata so downstream consumers can sanity-check. -----------
        "metadata": {
            "grid": "ieee24",
            "task": "binary",
            "split_config": asdict(cfg),
            "n_total": len(dataset),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "split_ratios_actual": {
                "train": len(train_idx) / len(dataset),
                "val": len(val_idx) / len(dataset),
                "test": len(test_idx) / len(dataset),
            },
            "stratified_on": "graph-level binary label (DNS > 0)",
            "generator": "sklearn.model_selection.train_test_split",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_SPLIT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    step(f"Wrote {OUT_SPLIT_JSON} "
         f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}).")


# --------------------------------------------------------------------------
# Phase 5 — Verify
# --------------------------------------------------------------------------

def verify_artefacts() -> None:
    """Sanity-check the three outputs without re-doing any training:
       1. The .pt loads via from_processed and yields the expected sample shape.
       2. The split JSON has the three required keys and indices in range.
       3. The checkpoint loads cleanly into a fresh GINe model and produces
          a forward pass on one batch.
    This is a lightweight stand-in for the full notebook dry-run on T4."""
    import torch  # type: ignore
    from torch_geometric.loader import DataLoader  # type: ignore

    # We deliberately import the FULL data module here (raw-aware), since
    # the prebuild box has it. The notebook uses a trimmed module with
    # `from_processed`; the .pt format is identical.
    from powergraph_data import PowerGraphDataset  # type: ignore
    from powergraph_models import GINe_Graph  # type: ignore

    banner("Phase 5 — Verifying outputs")

    # Dataset
    dataset = PowerGraphDataset(
        root=str(DATASET_ROOT),
        grid_name="ieee24",
        task="binary",
    )
    sample = dataset[0]
    step(f"Dataset OK: {len(dataset)} graphs, sample={sample}")
    assert sample.x.shape[1] == 3, f"node feature dim = {sample.x.shape[1]}, expected 3"
    assert sample.edge_attr.shape[1] == 5, (
        f"edge feature dim = {sample.edge_attr.shape[1]}, expected 5")

    # Split JSON
    with open(OUT_SPLIT_JSON) as f:
        split = json.load(f)
    for key in ("train", "val", "test"):
        if key not in split:
            raise AssertionError(f"split JSON missing required key '{key}'")
    n_split = len(split["train"]) + len(split["val"]) + len(split["test"])
    assert n_split == len(dataset), (
        f"split sum {n_split} != dataset size {len(dataset)}")
    all_idx = set(split["train"]) | set(split["val"]) | set(split["test"])
    assert all_idx == set(range(len(dataset))), "split indices have gaps/dupes"
    step(f"Split JSON OK: keys=['train','val','test'], "
         f"n_train={len(split['train'])}, "
         f"n_val={len(split['val'])}, n_test={len(split['test'])}")

    # Checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GINe_Graph(
        num_node_features=3,
        num_edge_features=5,
        hidden_dim=OPTUNA_BEST_GINE["hidden_dim"],
        num_layers=OPTUNA_BEST_GINE["num_layers"],
        num_classes=2,
        dropout=OPTUNA_BEST_GINE["dropout"],
        pooling=OPTUNA_BEST_GINE["pooling"],
    ).to(device)

    state = torch.load(OUT_CHECKPOINT, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=True)
    # `strict=True` raises if keys don't match; reaching here means clean load.
    model.eval()
    loader = DataLoader([dataset[i] for i in split["test"][:32]], batch_size=16)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            assert logits.shape[1] == 2
            break
    step(f"Checkpoint OK: loaded into GINe_Graph on {device}, forward pass produced logits {tuple(logits.shape)}.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="WM9B7 PowerGraph Azure pre-build (steps 1-4 of §5.4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Assume raw .mat files are already in "
                             f"{RAW_DIR} (e.g. you placed them manually).")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip phase 3 (training). Useful for re-running "
                             "the dataset/split steps when the checkpoint "
                             "already exists.")
    parser.add_argument("--force-train", action="store_true",
                        help="Re-train even if the checkpoint already exists.")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip phase 5 (verification).")
    args = parser.parse_args()

    print(f"\nWM9B7 PowerGraph Azure pre-build")
    print(f"  cwd:    {os.getcwd()}")
    print(f"  python: {sys.version.split()[0]}")
    try:
        import torch  # type: ignore
        cuda = torch.cuda.is_available()
        print(f"  torch:  {torch.__version__} | CUDA={cuda}"
              + (f" | device={torch.cuda.get_device_name(0)}" if cuda else ""))
    except Exception as exc:
        warn(f"torch not yet importable: {exc}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Phase 1 — raw .mat files ----------
    banner("Phase 1 — Raw data (Figshare → dataset_cascades/)")
    if args.skip_download:
        if not have_raw_files():
            raise SystemExit(
                f"--skip-download set but raw files are missing in {RAW_DIR}: "
                f"{[f for f in RAW_FILES if not (RAW_DIR / f).exists()]}"
            )
        step(f"Raw files present in {RAW_DIR}. Skipping download.")
    elif have_raw_files():
        step(f"Raw files already present in {RAW_DIR}. Skipping download.")
    else:
        tarball = Path(FIGSHARE_TARBALL_NAME)
        download_figshare(tarball)
        extract_ieee24_raw(tarball, RAW_DIR)
        # Free disk: the tarball is no longer needed once .mat files extracted.
        tarball.unlink()
        step(f"Removed {tarball} to reclaim disk.")

    # ---------- Phase 2 — process to .pt ----------
    banner("Phase 2 — Process raw .mat → PyG .pt")
    if PROCESSED_PT_SRC.exists() and OUT_DATASET.exists():
        step(f"{OUT_DATASET} already exists. Skipping processing.")
        dataset = None
    else:
        dataset = build_processed_dataset()
        copy_processed_pt()

    # ---------- Phase 3 — train ----------
    banner("Phase 3 — Train GINe (seed=23, Optuna-best HPs, 200 epochs)")
    if args.skip_train:
        warn("--skip-train set; phase 3 skipped.")
    else:
        train_gine(skip_if_exists=not args.force_train)

    # ---------- Phase 4 — split JSON ----------
    banner("Phase 4 — Persist deterministic split")
    write_split_json(dataset=dataset)

    # ---------- Phase 5 — verify ----------
    if not args.skip_verify:
        try:
            verify_artefacts()
        except AssertionError as exc:
            warn(f"Verification FAILED: {exc}")
            return 2

    # ---------- Done ----------
    banner("Done — outputs ready for download")
    for p in (OUT_DATASET, OUT_CHECKPOINT, OUT_SPLIT_JSON):
        ok = "OK " if p.exists() else "MISSING"
        size = fmt_bytes(p.stat().st_size) if p.exists() else "-"
        print(f"  [{ok}] {p}  ({size})")

    print("\nNext: download the contents of "
          f"{OUT_DIR}/ to your local machine, then run "
          "`python assemble_bundle.py` locally.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
