import json

import pytest

from iriscc.checkpoint_bundle import resolve_checkpoint_from_bundle


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
