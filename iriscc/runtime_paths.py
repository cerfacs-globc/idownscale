"""Shared runtime path resolution helpers for training, prediction, and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iriscc.checkpoint_bundle import activate_bundle_contract, resolve_checkpoint_from_bundle
from iriscc.settings import CONFIG, RUNS_DIR, get_evaluation_sample_dir


def resolve_checkpoint_path(
    exp: str,
    test_name: str,
    checkpoint_bundle: str | Path | None = None,
) -> Path:
    if checkpoint_bundle:
        activate_bundle_contract(checkpoint_bundle)
        return resolve_checkpoint_from_bundle(checkpoint_bundle)
    run_dir = RUNS_DIR / exp / test_name / "lightning_logs" / "version_best"
    return next((run_dir / "checkpoints").glob("best-checkpoint*.ckpt"))


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
