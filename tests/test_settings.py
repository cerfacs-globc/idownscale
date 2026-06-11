import os
import pytest

from iriscc.settings import CONFIG, DATASET_DIR, OUTPUT_DIR, REPO_DIR

def test_repo_dir_exists():
    assert REPO_DIR.exists()

def test_exp5_config():
    assert 'exp5' in CONFIG
    exp5 = CONFIG['exp5']
    assert exp5['fill_value'] == 0.0
    assert exp5['model'] == 'unet'

@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason="datasets directory not present in CI")
def test_dataset_dir_exists():
    assert DATASET_DIR.exists()
    assert DATASET_DIR == OUTPUT_DIR / "datasets"
