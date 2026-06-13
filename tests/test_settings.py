import os
import pytest

from iriscc.settings import CONFIG, DATASET_DIR, OUTPUT_DIR, REPO_DIR

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

@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="datasets directory not present in CI")
def test_dataset_dir_exists():
    assert DATASET_DIR.exists()
    assert DATASET_DIR == OUTPUT_DIR / "datasets"
