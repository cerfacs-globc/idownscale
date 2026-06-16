from pathlib import Path

import pytest

from iriscc import runtime_paths


def test_resolve_statistics_dir_prefers_hparams_override(tmp_path):
    sample_dir = tmp_path / "sample_dir"
    statistics_dir = tmp_path / "statistics_dir"
    hparams = {
        "sample_dir": sample_dir,
        "statistics_dir": statistics_dir,
    }
    assert runtime_paths.resolve_statistics_dir(hparams) == statistics_dir


def test_resolve_statistics_dir_falls_back_to_sample_dir(tmp_path):
    sample_dir = tmp_path / "sample_dir"
    hparams = {"sample_dir": sample_dir}
    assert runtime_paths.resolve_statistics_dir(hparams) == sample_dir


def test_resolve_runtime_sample_dir_prefers_explicit_override(tmp_path):
    explicit = tmp_path / "explicit_samples"
    resolved = runtime_paths.resolve_runtime_sample_dir(
        "exp5",
        "unet_all",
        explicit_sample_dir=explicit,
    )
    assert resolved == explicit


def test_resolve_runtime_sample_dir_uses_evaluation_mapping():
    resolved = runtime_paths.resolve_runtime_sample_dir("exp5", "gcm_raw", simu_test="gcm")
    assert resolved.name == "dataset_exp5_test_gcm"


def test_resolve_runtime_sample_dir_falls_back_to_hparams(tmp_path):
    sample_dir = tmp_path / "training_samples"
    resolved = runtime_paths.resolve_runtime_sample_dir(
        "exp5",
        "unet_all",
        hparams={"sample_dir": sample_dir},
    )
    assert resolved == sample_dir


def test_resolve_runtime_sample_dir_falls_back_to_config_dataset():
    resolved = runtime_paths.resolve_runtime_sample_dir("exp5", "unet_all")
    assert resolved == Path(runtime_paths.CONFIG["exp5"]["dataset"])


def test_resolve_checkpoint_path_uses_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_paths, "RUNS_DIR", tmp_path / "runs")
    checkpoint_dir = runtime_paths.RUNS_DIR / "exp5" / "unet_all" / "lightning_logs" / "version_best" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "best-checkpoint-epoch=01.ckpt"
    checkpoint.write_text("checkpoint")

    resolved = runtime_paths.resolve_checkpoint_path("exp5", "unet_all")
    assert resolved == checkpoint


def test_resolve_checkpoint_path_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_paths, "RUNS_DIR", tmp_path / "runs")
    checkpoint_dir = runtime_paths.RUNS_DIR / "exp5" / "unet_all" / "lightning_logs" / "version_best" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="best checkpoint"):
        runtime_paths.resolve_checkpoint_path("exp5", "unet_all")


def test_resolve_checkpoint_path_raises_when_ambiguous(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_paths, "RUNS_DIR", tmp_path / "runs")
    checkpoint_dir = runtime_paths.RUNS_DIR / "exp5" / "unet_all" / "lightning_logs" / "version_best" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best-checkpoint-epoch=01.ckpt").write_text("checkpoint")
    (checkpoint_dir / "best-checkpoint-epoch=02.ckpt").write_text("checkpoint")

    with pytest.raises(FileExistsError, match="Expected exactly one best checkpoint"):
        runtime_paths.resolve_checkpoint_path("exp5", "unet_all")


def test_resolve_checkpoint_path_uses_bundle_when_provided(monkeypatch, tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    called = {"activated": False}

    def fake_activate(path):
        called["activated"] = True
        assert Path(path) == bundle_dir

    def fake_resolve(path):
        assert Path(path) == bundle_dir
        return bundle_dir / "copied.ckpt"

    monkeypatch.setattr(runtime_paths, "activate_bundle_contract", fake_activate)
    monkeypatch.setattr(runtime_paths, "resolve_checkpoint_from_bundle", fake_resolve)

    resolved = runtime_paths.resolve_checkpoint_path("exp5", "unet_all", checkpoint_bundle=bundle_dir)
    assert called["activated"] is True
    assert resolved == bundle_dir / "copied.ckpt"


def test_require_match_allows_multiple_sorted(tmp_path):
    root = tmp_path / "samples"
    root.mkdir()
    later = root / "sample_20200102.npz"
    earlier = root / "sample_20200101.npz"
    later.write_text("later")
    earlier.write_text("earlier")

    resolved = runtime_paths.require_match(root, "sample_*.npz", "sample file", allow_multiple=True)
    assert resolved == [earlier, later]


def test_resolve_sample_file_returns_existing_path(tmp_path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    sample = sample_dir / "sample_20200101.npz"
    sample.write_text("sample")

    assert runtime_paths.resolve_sample_file(sample_dir, "20200101") == sample


def test_resolve_sample_file_raises_when_missing(tmp_path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="sample file for date 20200101"):
        runtime_paths.resolve_sample_file(sample_dir, "20200101")
