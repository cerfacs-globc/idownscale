from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def load_workflow_module():
    module_path = Path(__file__).resolve().parents[1] / "bin" / "production" / "run_exp5_workflow.py"
    spec = spec_from_file_location("run_exp5_workflow", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_list_phase1_outputs_requires_daily_frequency(monkeypatch, tmp_path):
    workflow = load_workflow_module()
    monkeypatch.setattr(workflow, "get_experiment_training_frequency", lambda exp: "3h")

    with pytest.raises(ValueError, match="expects daily outputs"):
        workflow.list_phase1_outputs("exp5", tmp_path, "20000101", "20000102")


def test_prediction_output_path_rejects_mixed_cadence(monkeypatch):
    import iriscc.settings as settings

    monkeypatch.setitem(settings.CONFIG["exp5"], "prediction_frequency", "3h")
    with pytest.raises(ValueError, match="mixed-cadence prediction is not implemented yet"):
        settings.get_prediction_output_path("exp5", "gcm_bc", "tas", "20000101", "20000102", "test_run")
