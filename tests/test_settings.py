import pytest
from pathlib import Path
from iriscc.settings import CONFIG, ROOT_DIR

def test_root_dir_exists():
    assert ROOT_DIR.exists()

def test_exp5_config():
    assert 'exp5' in CONFIG
    exp5 = CONFIG['exp5']
    assert exp5['fill_value'] == 0.0
    assert exp5['model'] == 'unet'

def test_dataset_dir_exists():
    dataset_dir = ROOT_DIR / 'datasets'
    assert dataset_dir.exists()
