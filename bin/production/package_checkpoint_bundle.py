#!/usr/bin/env python3
"""
Package a checkpoint together with the metadata and data-contract files needed
to understand and reuse it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "scratch" / "checkpoint_bundles"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"


class HParamsLoader(yaml.SafeLoader):
    pass


def _construct_python_tuple(loader: yaml.SafeLoader, node: yaml.Node) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node))


def _construct_posix_path(loader: yaml.SafeLoader, node: yaml.Node) -> Path:
    parts = loader.construct_sequence(node)
    return Path(*parts)


HParamsLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)
HParamsLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
    _construct_posix_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package an idownscale checkpoint bundle.")
    parser.add_argument("--exp", default="exp5", help="Experiment name, e.g. exp5.")
    parser.add_argument("--test-name", required=True, help="Run name, e.g. unet_all.")
    parser.add_argument(
        "--runs-root",
        default=os.getenv("IDOWNSCALE_RUNS_DIR", str(DEFAULT_RUNS_ROOT)),
        help="Root directory containing runs/<exp>/<test-name>/lightning_logs/version_best/.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where the bundle folder will be created.",
    )
    parser.add_argument("--copy-checkpoint", action="store_true", help="Copy the checkpoint file into the bundle.")
    return parser.parse_args()


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relink_candidate_paths(sample_dir: Path) -> list[Path]:
    basename = sample_dir.name
    candidates = [
        sample_dir,
        PROJECT_ROOT / "rawdata" / basename,
        PROJECT_ROOT / "scratch" / basename,
        Path("/gpfs-calypso/scratch/globc/page/idownscale_output/datasets") / basename,
        Path("/scratch/globc/page/idownscale_exp5/datasets") / basename,
        PROJECT_ROOT / "datasets" / basename,
    ]
    deduped: list[Path] = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def find_contract_files(sample_dir: Path) -> dict[str, Path | None]:
    wanted = ["statistics.json", "coordinates.npz", "gamma_params.npz"]
    contract: dict[str, Path | None] = {}
    candidates = relink_candidate_paths(sample_dir)
    for filename in wanted:
        contract[filename] = None
        for candidate_dir in candidates:
            candidate = candidate_dir / filename
            if candidate.exists():
                contract[filename] = candidate
                break
    return contract


def summarize_metrics(metrics_csv: Path) -> dict[str, float]:
    rows = []
    with metrics_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    summary: dict[str, float] = {}
    if not rows:
        return summary
    for key in ["loss", "rmse", "mae"]:
        values = [float(row[key]) for row in rows if row.get(key)]
        if values:
            summary[f"mean_{key}"] = sum(values) / len(values)
            summary[f"min_{key}"] = min(values)
            summary[f"max_{key}"] = max(values)
    summary["n_test_samples"] = float(len(rows))
    return summary


def copy_if_present(path: Path | None, dest_dir: Path) -> str | None:
    if path is None or not path.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / path.name
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
    return str(target)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.runs_root) / args.exp / args.test_name / "lightning_logs" / "version_best"
    checkpoint_matches = sorted((run_dir / "checkpoints").glob("best-checkpoint*.ckpt"))
    if not checkpoint_matches:
        raise FileNotFoundError(f"No best checkpoint found under {run_dir / 'checkpoints'}")

    checkpoint_path = checkpoint_matches[0]
    hparams_path = run_dir / "hparams.yaml"
    metrics_path = run_dir / "metrics_test_set.csv"
    if not hparams_path.exists():
        raise FileNotFoundError(f"Missing hparams file: {hparams_path}")

    with hparams_path.open() as handle:
        loaded = yaml.load(handle, Loader=HParamsLoader)
    hparams = loaded["hparams"] if isinstance(loaded, dict) and "hparams" in loaded else loaded
    sample_dir = Path(hparams["sample_dir"])
    contract_files = find_contract_files(sample_dir)

    bundle_dir = Path(args.output_root) / f"{args.exp}_{args.test_name}_bundle"
    metadata_dir = bundle_dir / "metadata"
    contract_dir = bundle_dir / "data_contract"
    checkpoint_dir = bundle_dir / "checkpoint"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hparams_path, metadata_dir / "hparams.yaml")
    if metrics_path.exists():
        shutil.copy2(metrics_path, metadata_dir / "metrics_test_set.csv")

    copied_contract = {
        name: copy_if_present(path, contract_dir)
        for name, path in contract_files.items()
    }

    if args.copy_checkpoint:
        copied_checkpoint = copy_if_present(checkpoint_path, checkpoint_dir)
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        copied_checkpoint = None

    manifest = {
        "bundle_name": f"{args.exp}_{args.test_name}_bundle",
        "experiment": args.exp,
        "test_name": args.test_name,
        "model": hparams.get("model"),
        "loss": hparams.get("loss"),
        "img_size": list(hparams.get("img_size", [])),
        "in_channels": hparams.get("in_channels"),
        "channels": hparams.get("channels"),
        "fill_value": hparams.get("fill_value"),
        "mask": hparams.get("mask"),
        "output_norm": hparams.get("output_norm"),
        "original_training_sample_dir": str(sample_dir),
        "sample_dir_candidates_checked": [str(p) for p in relink_candidate_paths(sample_dir)],
        "run_dir": str(run_dir),
        "checkpoint": {
            "original_path": str(checkpoint_path),
            "sha256": sha256sum(checkpoint_path),
            "copied_path": copied_checkpoint,
        },
        "metadata_files": {
            "hparams_yaml": str(metadata_dir / "hparams.yaml"),
            "metrics_test_set_csv": str(metadata_dir / "metrics_test_set.csv") if metrics_path.exists() else None,
        },
        "contract_files": {
            name: {
                "resolved_source": str(path) if path else None,
                "copied_path": copied_contract[name],
                "sha256": sha256sum(path) if path and path.exists() else None,
            }
            for name, path in contract_files.items()
        },
        "metrics_summary": summarize_metrics(metrics_path) if metrics_path.exists() else {},
        "notes": [
            "A checkpoint is only portable together with the preprocessing and normalization contract that defined its training world.",
            "If the reference reanalysis, target dataset, grid, predictors, or normalization change, retraining is usually required.",
            "The original hparams.yaml can contain legacy sample_dir or runs_dir paths from older filesystem layouts.",
        ],
    }

    manifest_path = bundle_dir / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    readme = f"""idownscale checkpoint bundle: {args.exp}/{args.test_name}

Contents
- checkpoint_manifest.json: machine-readable contract summary
- metadata/hparams.yaml: original Lightning hyperparameters saved with the run
- metadata/metrics_test_set.csv: historical test metrics saved by training
- data_contract/: resolved sidecar files such as statistics.json when available
- checkpoint/: copied checkpoint file when --copy-checkpoint is used

Important
- The checkpoint alone is not enough for trustworthy reuse.
- Keep predictor definitions, channel order, grids, masking, and normalization consistent.
- A new reanalysis or modified setup should trigger retraining, not blind reuse.
"""
    (bundle_dir / "README.txt").write_text(readme)
    print(bundle_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
