import os

import pytest

from iriscc.settings import (
    CONFIG,
    DATASET_BC_DIR,
    DATASET_DIR,
    OUTPUT_DIR,
    REPO_DIR,
    get_bc_bundle_path,
    get_bias_corrected_netcdf_path,
    get_bias_corrected_sample_dir,
    get_dataset_variant_dir,
    get_experiment_output_frequency,
    get_experiment_prediction_frequency,
    get_experiment_training_frequency,
    get_frequency_filename_token,
    require_matching_experiment_frequencies,
    get_source_aggregation_method,
    get_source_default_frequency,
    get_source_native_frequency,
)

def test_repo_dir_exists():
    assert REPO_DIR.exists()

def test_exp5_config():
    assert "exp5" in CONFIG
    exp5 = CONFIG["exp5"]
    assert exp5["fill_value"] == 0.0
    assert exp5["model"] == "unet"


def test_expc_config():
    assert "expc" in CONFIG
    expc = CONFIG["expc"]
    assert expc["target"] == "cerra"
    assert expc["target_source"] == "cerra"
    assert expc["fill_value"] == 0.0
    assert expc["shape"] == (503, 326)
    assert expc["phase1_target_method"] == "bilinear"
    assert expc["phase1_crop_target"] is False


def test_cerra_frequency_metadata_is_explicit():
    assert get_source_native_frequency("cerra") == "3h"
    assert get_source_default_frequency("cerra") == "daily"
    assert get_source_aggregation_method("cerra", "daily") == "mean"


def test_active_temperature_workflows_resolve_daily_training_and_prediction_frequency():
    assert get_experiment_training_frequency("exp5") == "daily"
    assert get_experiment_prediction_frequency("exp5") == "daily"
    assert get_experiment_output_frequency("exp5") == "daily"
    assert get_experiment_training_frequency("expc") == "daily"
    assert get_experiment_prediction_frequency("expc") == "daily"
    assert get_frequency_filename_token(get_experiment_prediction_frequency("exp5")) == "day"


def test_matching_frequencies_are_required_for_current_runtime():
    assert require_matching_experiment_frequencies("exp5") == "daily"


def test_bc_bundle_paths_are_experiment_specific():
    exp5_path = get_bc_bundle_path("exp5", "gcm", "train_hist")
    expc_path = get_bc_bundle_path("expc", "gcm", "train_hist")
    assert exp5_path == DATASET_BC_DIR / "bc_train_hist_exp5_gcm.npz"
    assert expc_path == DATASET_BC_DIR / "bc_train_hist_expc_gcm.npz"
    assert exp5_path != expc_path


def test_bias_corrected_sample_dirs_are_experiment_specific():
    exp5_dir = get_bias_corrected_sample_dir("exp5", "gcm")
    expc_dir = get_bias_corrected_sample_dir("expc", "gcm")
    assert exp5_dir == DATASET_BC_DIR / "dataset_exp5_test_gcm_bc"
    assert expc_dir == DATASET_BC_DIR / "dataset_expc_test_gcm_bc"
    assert exp5_dir != expc_dir


def test_bias_corrected_sample_dirs_include_bc_tag():
    tagged_dir = get_bias_corrected_sample_dir("expc", "gcm", bc_tag="output norm")
    assert tagged_dir == DATASET_BC_DIR / "dataset_expc_test_gcm_bc_output_norm"


def test_dataset_variant_dir_is_experiment_specific():
    exp5_dir = get_dataset_variant_dir("exp5", "gcm")
    expc_dir = get_dataset_variant_dir("expc", "gcm")
    assert exp5_dir == DATASET_BC_DIR / "dataset_exp5_test_gcm"
    assert expc_dir == DATASET_BC_DIR / "dataset_expc_test_gcm"
    assert exp5_dir != expc_dir


def test_bias_corrected_netcdf_names_encode_experiment_windows():
    exp5_path = get_bias_corrected_netcdf_path("exp5", "gcm", "tas", "test_hist")
    expc_path = get_bias_corrected_netcdf_path("expc", "gcm", "tas", "test_hist")
    assert "_day_" in exp5_path.name
    assert "_day_" in expc_path.name
    assert "20000101-20141231" in exp5_path.name
    assert "20000101-20210910" in expc_path.name
    assert exp5_path.name != expc_path.name

@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="datasets directory not present in CI")
def test_dataset_dir_exists():
    assert DATASET_DIR.exists()
    assert DATASET_DIR == OUTPUT_DIR / "datasets"
