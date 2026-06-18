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
    local_matches = sorted((bundle_dir / "checkpoint").glob("*.ckpt"))
    if len(local_matches) == 1:
        return local_matches[0]
    if len(local_matches) > 1:
        joined = ", ".join(path.name for path in local_matches)
        message = f"Expected exactly one bundled checkpoint under {bundle_dir / 'checkpoint'}, found {len(local_matches)}: {joined}"
        raise FileExistsError(message)
    message = f"No usable checkpoint found for bundle {bundle_dir}"
    raise FileNotFoundError(message)


def activate_bundle_contract(bundle_dir: str | Path) -> dict[str, Any]:
    """Export environment hints for bundle-aware metadata resolution.

    Existing transform resolution logic can use the bundle metadata without
    requiring every caller to reimplement path logic.
    """
    bundle_dir = Path(bundle_dir)
    manifest = load_bundle_manifest(bundle_dir)
    stats_catalog = manifest.get("bundle_files", manifest.get("contract_files", {}))
    stats_entry = stats_catalog.get("statistics.json", {})
    copied_stats = stats_entry.get("copied_path")
    resolved_stats = stats_entry.get("resolved_source")
    local_stats = bundle_dir / "data_contract" / "statistics.json"
    stats_dir = None
    if copied_stats and Path(copied_stats).exists():
        stats_dir = str(Path(copied_stats).parent)
    elif resolved_stats and Path(resolved_stats).exists():
        stats_dir = str(Path(resolved_stats).parent)
    elif local_stats.exists():
        stats_dir = str(local_stats.parent)
    if stats_dir:
        os.environ["IDOWNSCALE_SAMPLE_STATS_DIR"] = stats_dir
        os.environ["IDOWNSCALE_ALLOW_STATISTICS_FALLBACK"] = "1"
    return manifest
