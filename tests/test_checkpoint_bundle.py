import json

import os

import pytest

from iriscc.checkpoint_bundle import activate_bundle_contract, resolve_checkpoint_from_bundle


def test_resolve_checkpoint_from_bundle_falls_back_to_local_checkpoint(tmp_path):
    bundle_dir = tmp_path / "bundle"
    checkpoint_dir = bundle_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "best-checkpoint.ckpt"
    checkpoint.write_text("checkpoint")
    manifest = {
        "checkpoint": {
            "copied_path": "/nonexistent/copied.ckpt",
            "original_path": "/nonexistent/original.ckpt",
        }
    }
    (bundle_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest))

    assert resolve_checkpoint_from_bundle(bundle_dir) == checkpoint


def test_resolve_checkpoint_from_bundle_raises_when_local_checkpoint_is_ambiguous(tmp_path):
    bundle_dir = tmp_path / "bundle"
    checkpoint_dir = bundle_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "a.ckpt").write_text("a")
    (checkpoint_dir / "b.ckpt").write_text("b")
    manifest = {"checkpoint": {"copied_path": "/nonexistent/copied.ckpt"}}
    (bundle_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(FileExistsError, match="Expected exactly one bundled checkpoint"):
        resolve_checkpoint_from_bundle(bundle_dir)


def test_activate_bundle_contract_falls_back_to_local_statistics(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "bundle"
    data_contract_dir = bundle_dir / "data_contract"
    data_contract_dir.mkdir(parents=True)
    (data_contract_dir / "statistics.json").write_text("{}")
    manifest = {"contract_files": {"statistics.json": {"copied_path": "/nonexistent/statistics.json"}}}
    (bundle_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest))

    monkeypatch.delenv("IDOWNSCALE_SAMPLE_STATS_DIR", raising=False)
    monkeypatch.delenv("IDOWNSCALE_ALLOW_STATISTICS_FALLBACK", raising=False)

    activate_bundle_contract(bundle_dir)

    assert os.environ["IDOWNSCALE_SAMPLE_STATS_DIR"] == str(data_contract_dir)
    assert os.environ["IDOWNSCALE_ALLOW_STATISTICS_FALLBACK"] == "1"
