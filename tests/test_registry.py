import pytest
from iriscc.registry import get_debiaser, get_model
import torch

def test_get_debiaser_ibicus():
    # ibicus should be installed in the environment
    try:
        cdft = get_debiaser('cdft')
        assert cdft.__name__ == 'CDFt'
    except ImportError:
        pytest.skip("Ibicus not installed")

def test_get_debiaser_invalid():
    with pytest.raises(ValueError, match="could not be found"):
        get_debiaser('invalid_debiaser_name_123')

def test_get_model_unet():
    unet_class = get_model('unet')
    assert unet_class.__name__ == 'UNet'
    # Smoke test instantiation
    # UNet(n_channels=3, n_classes=1) -> but registry returns the class
    pass

def test_get_model_swin2sr():
    swin_class = get_model('swin2sr')
    assert swin_class.__name__ == 'Swin2SR'

def test_get_model_invalid():
    with pytest.raises(ValueError, match="is not registered"):
        get_model('invalid_model_name_123')
