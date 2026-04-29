"""
Helpers for loading portable checkpoint bundles.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_bundle_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "checkpoint_manifest.json"
    with manifest_path.open() as handle:
        return json.load(handle)


def resolve_checkpoint_from_bundle(bundle_dir: str | Path) -> Path:
    bundle_dir = Path(bundle_dir)
    manifest = load_bundle_manifest(bundle_dir)
    copied = manifest.get("checkpoint", {}).get("copied_path")
    original = manifest.get("checkpoint", {}).get("original_path")
    if copied and Path(copied).exists():
        return Path(copied)
    if original and Path(original).exists():
        return Path(original)
    message = f"No usable checkpoint found for bundle {bundle_dir}"
    raise FileNotFoundError(message)


def activate_bundle_contract(bundle_dir: str | Path) -> dict[str, Any]:
    """

    Export environment hints so existing transform resolution logic can use the
    bundle contract without requiring every caller to reimplement path logic.
    """
    bundle_dir = Path(bundle_dir)
    manifest = load_bundle_manifest(bundle_dir)
    stats_entry = manifest.get("contract_files", {}).get("statistics.json", {})
    copied_stats = stats_entry.get("copied_path")
    resolved_stats = stats_entry.get("resolved_source")
    stats_dir = None
    if copied_stats and Path(copied_stats).exists():
        stats_dir = str(Path(copied_stats).parent)
    elif resolved_stats and Path(resolved_stats).exists():
        stats_dir = str(Path(resolved_stats).parent)
    if stats_dir:
        os.environ["IDOWNSCALE_SAMPLE_STATS_DIR"] = stats_dir
    return manifest
