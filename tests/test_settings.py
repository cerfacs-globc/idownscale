import pytest
from pathlib import Path
from iriscc.settings import CONFIG, REPO_DIR

def test_repo_dir_exists():
    assert REPO_DIR.exists()

def test_exp5_config():
    assert 'exp5' in CONFIG
    exp5 = CONFIG['exp5']
    assert exp5['fill_value'] == 0.0
    assert exp5['model'] == 'unet'

def test_dataset_dir_exists():
    # datasets directory is expected in the repo root
    dataset_dir = REPO_DIR / 'datasets'
    assert dataset_dir.exists()
