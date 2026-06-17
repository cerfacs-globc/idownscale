"""Shared runtime path resolution helpers for training, prediction, and evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from iriscc.checkpoint_bundle import activate_bundle_contract, resolve_checkpoint_from_bundle
from iriscc.settings import (
    CONFIG,
    DATASET_DIR,
    EXP5_ARCHIVE_DATASET_DIR,
    RUNS_DIR,
    format_sample_time_token,
    get_evaluation_sample_dir,
)


def require_existing_file(path: str | Path, description: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        message = f"Missing {description}: {resolved}"
        raise FileNotFoundError(message)
    if not resolved.is_file():
        message = f"Expected {description} to be a file, found: {resolved}"
        raise FileNotFoundError(message)
    return resolved


def require_match(
    root: str | Path,
    pattern: str,
    description: str,
    *,
    allow_multiple: bool = False,
) -> Path | list[Path]:
    root_path = Path(root)
    matches = sorted(root_path.glob(pattern))
    if not matches:
        message = f"No {description} found under {root_path} matching {pattern}"
        raise FileNotFoundError(message)
    if allow_multiple:
        return matches
    if len(matches) > 1:
        joined = ", ".join(str(path.name) for path in matches)
        message = (
            f"Expected exactly one {description} under {root_path} matching {pattern}, "
            f"found {len(matches)}: {joined}"
        )
        raise FileExistsError(message)
    return matches[0]


def resolve_sample_file(sample_dir: str | Path, date_token: str) -> Path:
    return require_existing_file(
        Path(sample_dir) / f"sample_{date_token}.npz",
        f"sample file for date {date_token}",
    )


def resolve_sample_file_for_timestamp(sample_dir: str | Path, timestamp, frequency: str) -> Path:
    return resolve_sample_file(sample_dir, format_sample_time_token(timestamp, frequency))


def resolve_checkpoint_path(
    exp: str,
    test_name: str,
    checkpoint_bundle: str | Path | None = None,
) -> Path:
    if checkpoint_bundle:
        activate_bundle_contract(checkpoint_bundle)
        return resolve_checkpoint_from_bundle(checkpoint_bundle)
    run_dir = RUNS_DIR / exp / test_name / "lightning_logs" / "version_best"
    return require_match(run_dir / "checkpoints", "best-checkpoint*.ckpt", "best checkpoint")


def resolve_statistics_sample_dir(sample_dir: str | Path) -> Path:
    """
    Resolve a usable directory containing ``statistics.json``.

    Archival checkpoints can carry stale absolute sample_dir values from older
    filesystem layouts. For inference and evaluation we only need the matching
    statistics file, so fall back through a small set of compatible locations.
    """
    sample_dir = Path(sample_dir)
    if (sample_dir / "statistics.json").exists():
        return sample_dir

    allow_fallback = os.getenv("IDOWNSCALE_ALLOW_STATISTICS_FALLBACK", "").lower() in {"1", "true", "yes", "on"}
    if not allow_fallback:
        message = (
            f"Missing statistics.json in {sample_dir}. "
            "Compute dataset/run statistics first, or set IDOWNSCALE_ALLOW_STATISTICS_FALLBACK=1 "
            "only for explicit archival compatibility."
        )
        raise FileNotFoundError(message)

    candidates = [sample_dir]
    env_override = os.getenv("IDOWNSCALE_SAMPLE_STATS_DIR")
    if env_override:
        candidates.append(Path(env_override))
    candidates.append(DATASET_DIR / sample_dir.name)
    candidates.append(EXP5_ARCHIVE_DATASET_DIR)

    seen = set()
    for candidate in candidates:
        expanded_candidate = candidate.expanduser()
        if expanded_candidate in seen:
            continue
        seen.add(expanded_candidate)
        if (expanded_candidate / "statistics.json").exists():
            print(f"[warn] using fallback statistics directory {expanded_candidate} for {sample_dir}", flush=True)
            return expanded_candidate

    message = f"No statistics.json found for {sample_dir} or fallback candidates."
    raise FileNotFoundError(message)


def resolve_statistics_dir(hparams: dict[str, Any]) -> Path:
    return resolve_statistics_sample_dir(hparams.get("statistics_dir", hparams["sample_dir"]))


def resolve_first_sample_file(sample_dir: str | Path) -> Path:
    matches = require_match(sample_dir, "sample_*.npz", "sample file", allow_multiple=True)
    return matches[0]


def resolve_runtime_sample_dir(
    exp: str,
    test_name: str,
    *,
    simu_test: str | None = None,
    hparams: dict[str, Any] | None = None,
    explicit_sample_dir: str | Path | None = None,
) -> Path:
    if explicit_sample_dir is not None:
        return Path(explicit_sample_dir)

    resolved = get_evaluation_sample_dir(exp, test_name, simu_test)
    if resolved is not None:
        return Path(resolved)

    if hparams is not None and "sample_dir" in hparams:
        return Path(hparams["sample_dir"])

    return Path(CONFIG[exp]["dataset"])
