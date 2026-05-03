"""WM9B7 PowerGraph — local bundle assembly script.

Implements §5.4 steps 5-7: take the three Azure-produced artefacts, combine
them with the local results JSONs and figure PNGs, write a *deterministic*
`submission_bundle.tar.gz`, and print the SHA-256 ready to paste into the
notebook.

Determinism note:
    `tarfile.open("w:gz")` uses gzip's default header, which embeds the
    current timestamp — so two runs over identical inputs produce different
    bytes and different SHA-256s. We sidestep that by writing a plain `.tar`
    first (with all member metadata zeroed and entries sorted), then
    re-gzipping with `gzip.GzipFile(mtime=0)`. The resulting tarball is
    byte-for-byte reproducible across machines.

Usage:
    python assemble_bundle.py \\
        --azure-out      ./prebuild_outputs \\
        --results-dir    ./artifacts/results \\
        --figures-dir    ./artifacts/figures \\
        --output         ./submission_bundle.tar.gz

If your local layout matches the project tree (i.e. the JSONs and PNGs all
sit under one umbrella `artifacts/` directory), you can pass `--artifacts-dir`
instead and let the script find each file by name.

The script is strict about file membership: if a JSON or PNG named in the
manifest is missing, it errors out — better than silently shipping an
incomplete bundle the notebook will then crash on.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tarfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------
# Manifest — what goes in the bundle.
# Keep aligned with §5.2 of the build plan and the notebook's reads.
# --------------------------------------------------------------------------

# Files at the bundle root.
ROOT_FILES = [
    "ieee24_processed.pt",
    "ieee24_gine_best.pt",
    "ieee24_split_indices.json",
]

# results/*.json — every JSON the notebook (or the report appendix) cites.
RESULTS_JSONS = [
    "architecture_comparison_summary.json",
    "ml_baselines_summary.json",
    "robustness_summary.json",
    "conformal_summary.json",
    "mc_dropout_summary.json",
    "learning_curve_summary.json",
    "feature_ablation_summary.json",
    "pooling_ablation_summary.json",
    "explainability_summary.json",
    "threshold_optimisation_summary.json",
    "ensemble_summary.json",
    "shap_importance_concatenated.json",
    "multiclass_summary.json",
    "experiment_18_comparison.json",
    "experiment_20_simgrace.json",
    "dns_regression_summary.json",
    "edge_level_results.json",
    "cf_summary.json",
    "cross_grid_results.json",
    "onnx_benchmark_results.json",
    "torch_compile_benchmark.json",
    "ieee24_validation_report.json",
    "eda_summary.json",
]

# figures/*.png — every figure the notebook displays (or the report references
# in its appendix). The list is intentionally inclusive: the bundle is small
# (<60 MB) so we err on the side of completeness rather than risk a missing
# image at marker-time.
FIGURE_PNGS = [
    # EDA
    "01_label_distributions.png",
    "02_node_features.png",
    "03_edge_features.png",
    "04_line_loading.png",
    "05_node_features_by_class.png",
    "06_edges_tripped.png",
    "07_vulnerable_edges.png",
    "08_feature_correlations.png",
    "09_grid_topology.png",
    # Notebook cell 12 — the evidence gallery
    "topology_degradation.png",
    "feature_noise_degradation.png",
    "conformal_prediction.png",
    "learning_curve.png",
    "deployment_latency_comparison.png",
    "simgrace_tsne_comparison.png",
    "topk_sweep.png",
    "mc_dropout_uncertainty.png",
    "feature_ablation_drop.png",
    # Report-appendix figures (cited but not displayed in cell 12)
    "cf_vs_ig_gt_overlap.png",
    "cross_grid_roc.png",
    "cross_grid_calibration.png",
    "throughput_scaling.png",
    "dns_ablation_comparison.png",
    "edge_pr_curve_standard.png",
    "edge_pr_curve_network_centric.png",
    "edge_failure_topology.png",
    "shap_top20_concatenated.png",
    "threshold_optimisation.png",
    "threshold_cost_analysis.png",
    "simgrace_pretrain_loss.png",
]


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------

def find_file(name: str, search_dirs: List[Path]) -> Optional[Path]:
    """Return the first hit for `name` across search_dirs (recursive). None
    if nothing matches. Recursive walk lets us be tolerant of users who
    keep their JSONs under e.g. artifacts/results/ vs artifacts/eda/ vs
    artifacts/training/<arch>/."""
    for root in search_dirs:
        if not root.exists():
            continue
        # Direct hit first (fastest path).
        direct = root / name
        if direct.is_file():
            return direct
        # Recursive search.
        for hit in root.rglob(name):
            if hit.is_file():
                return hit
    return None


def resolve_inputs(args) -> Tuple[List[Tuple[Path, str]], List[str]]:
    """Build a (source_path, archive_path) list and a list of missing files."""
    azure_out = Path(args.azure_out).resolve()

    # The user can either point at one umbrella artifacts dir, or split
    # results-dir / figures-dir explicitly.
    results_search: List[Path] = []
    figures_search: List[Path] = []
    if args.artifacts_dir:
        root = Path(args.artifacts_dir).resolve()
        results_search.append(root)
        figures_search.append(root)
    if args.results_dir:
        results_search.append(Path(args.results_dir).resolve())
    if args.figures_dir:
        figures_search.append(Path(args.figures_dir).resolve())
    if not results_search:
        results_search.append(Path.cwd())
    if not figures_search:
        figures_search.append(Path.cwd())

    plan: List[Tuple[Path, str]] = []
    missing: List[str] = []

    for name in ROOT_FILES:
        src = azure_out / name
        if not src.is_file():
            missing.append(f"{name} (expected at {src})")
            continue
        plan.append((src, name))

    for name in RESULTS_JSONS:
        hit = find_file(name, results_search)
        if hit is None:
            missing.append(f"results/{name} (searched: "
                           f"{[str(p) for p in results_search]})")
            continue
        plan.append((hit, f"results/{name}"))

    for name in FIGURE_PNGS:
        hit = find_file(name, figures_search)
        if hit is None:
            missing.append(f"figures/{name} (searched: "
                           f"{[str(p) for p in figures_search]})")
            continue
        plan.append((hit, f"figures/{name}"))

    # Sort by archive path for deterministic ordering inside the tar.
    plan.sort(key=lambda pair: pair[1])
    return plan, missing


# --------------------------------------------------------------------------
# Deterministic tar + gzip
# --------------------------------------------------------------------------

def write_deterministic_tar_gz(plan: List[Tuple[Path, str]], output: Path) -> None:
    """Two-step build:
       1. Stream a plain tar to memory with all metadata zeroed and entries
          in sorted order.
       2. Gzip the tar bytes with mtime=0, max compression, no filename in
          the gzip header.
    The result is byte-identical across machines and runs."""
    # Step 1 — build a plain tar in a BytesIO buffer.
    tar_buf = BytesIO()
    # USTAR format keeps the header simple; PAX would embed extended headers
    # whose ordering can introduce non-determinism on some Python versions.
    with tarfile.open(fileobj=tar_buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for src, arcname in plan:
            info = tarfile.TarInfo(name=arcname)
            info.size = src.stat().st_size
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.type = tarfile.REGTYPE
            with open(src, "rb") as f:
                tar.addfile(info, f)
    tar_bytes = tar_buf.getvalue()

    # Step 2 — gzip with mtime=0 (otherwise gzip embeds the wall clock).
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9,
                           fileobj=raw, mtime=0) as gz:
            gz.write(tar_bytes)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble the deterministic submission bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--azure-out", default="./prebuild_outputs",
                        help="Directory holding the three Azure-produced files "
                             "(default: ./prebuild_outputs).")
    parser.add_argument("--artifacts-dir", default=None,
                        help="Single umbrella directory under which all "
                             "results JSONs and figure PNGs can be found "
                             "(recursive search). Use this OR the more "
                             "specific --results-dir / --figures-dir.")
    parser.add_argument("--results-dir", default=None,
                        help="Directory containing results *.json files "
                             "(recursive search).")
    parser.add_argument("--figures-dir", default=None,
                        help="Directory containing figure *.png files "
                             "(recursive search).")
    parser.add_argument("--output", default="./submission_bundle.tar.gz",
                        help="Output tarball path "
                             "(default: ./submission_bundle.tar.gz).")
    parser.add_argument("--ignore-missing", action="store_true",
                        help="Continue building the bundle even if some "
                             "manifest entries are not found locally. The "
                             "notebook will crash on any missing entry, so "
                             "use this only for debugging.")
    parser.add_argument("--print-manifest-only", action="store_true",
                        help="Resolve sources and print the inclusion plan "
                             "without writing the tarball.")
    args = parser.parse_args()

    print("WM9B7 PowerGraph — bundle assembly\n")

    plan, missing = resolve_inputs(args)

    print(f"Resolved {len(plan)} entries:")
    for src, arc in plan:
        size = src.stat().st_size
        print(f"  {arc:<50}  ←  {src}  ({size:,} B)")

    if missing:
        print(f"\n{len(missing)} entries could not be resolved:")
        for m in missing:
            print(f"  MISSING: {m}")
        if not args.ignore_missing:
            print("\nAborting. Pass --ignore-missing to build a partial bundle "
                  "(NOT recommended).")
            return 2

    if args.print_manifest_only:
        return 0

    output = Path(args.output).resolve()
    print(f"\nWriting deterministic bundle → {output}")
    write_deterministic_tar_gz(plan, output)
    sha = sha256_of(output)

    print(f"\nBundle:    {output}")
    print(f"Size:      {output.stat().st_size:,} B "
          f"({output.stat().st_size / (1024*1024):.1f} MB)")
    print(f"SHA-256:   {sha}")

    print("\nPaste these into the notebook (cell 7, the BUNDLE_URL / "
          "EXPECTED_SHA256 placeholders):\n")
    print(f"    BUNDLE_URL = \"https://github.com/<USERNAME>/"
          f"wm9b7-powergraph-submission/releases/download/v1.0/"
          f"{output.name}\"")
    print(f"    EXPECTED_SHA256 = \"{sha}\"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
