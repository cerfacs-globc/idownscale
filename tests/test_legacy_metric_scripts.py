from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = spec_from_file_location(module_name, module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("module_name", "relative_path"),
    [
        ("compute_test_metrics_day_rcm", "bin/evaluation/compute_test_metrics_day_rcm.py"),
        ("compute_test_metrics_month_rcm", "bin/evaluation/compute_test_metrics_month_rcm.py"),
    ],
)
def test_resolve_rcm_target_file_returns_unique_match(tmp_path, monkeypatch, module_name, relative_path):
    module = load_module(module_name, relative_path)
    monkeypatch.setattr(module, "RCM_RAW_DIR", tmp_path / "rcm_raw")
    target_dir = module.RCM_RAW_DIR / "ALADIN_reformat"
    target_dir.mkdir(parents=True)
    target_file = target_dir / "tas_example.nc"
    target_file.write_text("stub")

    assert module.resolve_rcm_target_file("tas") == target_file


@pytest.mark.parametrize(
    ("module_name", "relative_path", "error_type"),
    [
        ("compute_test_metrics_day_rcm", "bin/evaluation/compute_test_metrics_day_rcm.py", FileNotFoundError),
        ("compute_test_metrics_month_rcm", "bin/evaluation/compute_test_metrics_month_rcm.py", FileNotFoundError),
    ],
)
def test_resolve_rcm_target_file_raises_when_missing(tmp_path, monkeypatch, module_name, relative_path, error_type):
    module = load_module(module_name, relative_path)
    monkeypatch.setattr(module, "RCM_RAW_DIR", tmp_path / "rcm_raw")
    (module.RCM_RAW_DIR / "ALADIN_reformat").mkdir(parents=True)

    with pytest.raises(error_type, match="RCM target file for tas"):
        module.resolve_rcm_target_file("tas")


@pytest.mark.parametrize(
    ("module_name", "relative_path"),
    [
        ("compute_test_metrics_day_rcm", "bin/evaluation/compute_test_metrics_day_rcm.py"),
        ("compute_test_metrics_month_rcm", "bin/evaluation/compute_test_metrics_month_rcm.py"),
    ],
)
def test_resolve_rcm_target_file_raises_when_ambiguous(tmp_path, monkeypatch, module_name, relative_path):
    module = load_module(module_name, relative_path)
    monkeypatch.setattr(module, "RCM_RAW_DIR", tmp_path / "rcm_raw")
    target_dir = module.RCM_RAW_DIR / "ALADIN_reformat"
    target_dir.mkdir(parents=True)
    (target_dir / "tas_a.nc").write_text("a")
    (target_dir / "tas_b.nc").write_text("b")

    with pytest.raises(FileExistsError, match="RCM target file for tas"):
        module.resolve_rcm_target_file("tas")
