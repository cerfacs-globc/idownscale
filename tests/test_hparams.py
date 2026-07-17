from iriscc.hparams import IRISCCHyperParameters


def test_hparams_initialization():
    hparams = IRISCCHyperParameters()
    assert hparams.fill_value == 0.0
    assert hparams.loss == "masked_mse"
    assert hparams.exp == "exp5/unet_all"


def test_hparams_sample_dir_override(tmp_path):
    hparams = IRISCCHyperParameters(sample_dir=tmp_path)
    assert hparams.sample_dir == tmp_path
    assert hparams.statistics_dir == tmp_path


def test_hparams_statistics_dir_override(tmp_path):
    sample_dir = tmp_path / "samples"
    stats_dir = tmp_path / "stats"
    hparams = IRISCCHyperParameters(sample_dir=sample_dir, statistics_dir=stats_dir)
    assert hparams.sample_dir == sample_dir
    assert hparams.statistics_dir == stats_dir
