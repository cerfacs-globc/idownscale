from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest
from iriscc import settings


def load_workflow_module():
    module_path = Path(__file__).resolve().parents[1] / "bin" / "production" / "run_obs_workflow.py"
    spec = spec_from_file_location("run_obs_workflow", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_list_phase1_outputs_requires_daily_frequency(monkeypatch, tmp_path):
    workflow = load_workflow_module()
    monkeypatch.setattr(workflow, "get_experiment_training_frequency", lambda exp: "3h")

    with pytest.raises(ValueError, match="expects daily outputs"):
        workflow.list_phase1_outputs("exp5", tmp_path, "20000101", "20000102")


def test_prediction_output_path_accepts_fixed_step_mixed_cadence(monkeypatch):
    monkeypatch.setitem(settings.CONFIG["exp5"], "prediction_frequency", "3h")
    path = settings.get_prediction_output_path("exp5", "gcm_bc", "tas", "20000101", "20000102", "test_run")
    assert "_3h_" in path.name


def test_prediction_output_path_matches_metrics_default_naming():
    metrics_test_name = settings.get_metrics_test_name("unet_all", "gcm_bc")
    path = settings.get_prediction_output_path(
        "exp5",
        "gcm_bc",
        "tas",
        "20000101",
        "20141231",
        metrics_test_name,
    )
    assert path.name.endswith("_20000101_20141231_exp5_unet_all_gcm_bc.nc")


def test_default_comparison_step_is_added_for_evaluation_steps():
    workflow = load_workflow_module()

    class Args:
        skip_default_comparisons = False

    resolved = workflow.maybe_add_default_comparison_step(["predict_loop", "metrics_day"], Args())
    assert resolved[-1] == "compare_suite"


def test_default_comparison_step_can_be_disabled():
    workflow = load_workflow_module()

    class Args:
        skip_default_comparisons = True

    resolved = workflow.maybe_add_default_comparison_step(["predict_loop", "metrics_day"], Args())
    assert "compare_suite" not in resolved


def test_sbck_mbcn_requires_two_paired_variables():
    workflow = load_workflow_module()
    argv = [
        "run_obs_workflow.py",
        "--steps",
        "bc_dataset,bc_apply",
        "--bc-method",
        "sbck_mbcn",
        "--paired-vars",
        "uas",
    ]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(ValueError, match="exactly two variables"):
            workflow.main()


def test_sbck_mbcn_rejects_scalar_ml_steps():
    workflow = load_workflow_module()
    argv = [
        "run_obs_workflow.py",
        "--steps",
        "bc_dataset,bc_apply,pp_dataset",
        "--bc-method",
        "sbck_mbcn",
        "--paired-vars",
        "uas,vas",
    ]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(NotImplementedError, match="require deriving a scalar product first"):
            workflow.main()


def test_sbck_mbcn_allows_speed_scalar_downstream_after_derivation():
    workflow = load_workflow_module()
    argv = [
        "run_obs_workflow.py",
        "--steps",
        "bc_dataset,bc_apply,derive_products,pp_dataset",
        "--bc-method",
        "sbck_mbcn",
        "--paired-vars",
        "uas,vas",
        "--derive-wind-products",
        "--var",
        "sfcWind",
        "--dry-run",
    ]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "argv", argv)
        assert workflow.main() == 0


def test_sbck_mbcn_rejects_direction_scalar_metrics():
    workflow = load_workflow_module()
    argv = [
        "run_obs_workflow.py",
        "--steps",
        "bc_dataset,bc_apply,derive_products,metrics_day",
        "--bc-method",
        "sbck_mbcn",
        "--paired-vars",
        "uas,vas",
        "--derive-wind-products",
        "--var",
        "windFromDirection",
        "--test-name",
        "demo",
    ]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(NotImplementedError, match="circular diagnostics"):
            workflow.main()
