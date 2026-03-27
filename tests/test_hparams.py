import pytest
from iriscc.hparams import IRISCCHyperParameters

def test_hparams_initialization():
    hparams = IRISCCHyperParameters()
    assert hparams.fill_value == 0.0
    assert hparams.loss == 'masked_mse'
    assert hparams.exp == 'exp5/unet_all'
