"""Shared runtime path resolution helpers for training, prediction, and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iriscc.checkpoint_bundle import activate_bundle_contract, resolve_checkpoint_from_bundle
from iriscc.settings import CONFIG, RUNS_DIR, format_sample_time_token, get_evaluation_sample_dir


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


def resolve_statistics_dir(hparams: dict[str, Any]) -> Path:
    return Path(hparams.get("statistics_dir", hparams["sample_dir"]))


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
