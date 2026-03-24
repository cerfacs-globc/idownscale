import json

import torch

from iriscc.transforms import (
    Log10Transform,
    MinMaxNormalisation,
    Pad,
    StandardNormalisation,
    UnPad,
)

def test_standard_normalisation(tmp_path):
    # Mock statistics.json
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    stats_file = stats_dir / "statistics.json"
    with stats_file.open("w") as f:
        json.dump({"chan1": {"mean": 10.0, "std": 2.0}, "chan2": {"mean": 5.0, "std": 1.0}}, f)
    
    norm = StandardNormalisation(stats_dir)
    x = torch.tensor([[[12.0, 8.0]], [[6.0, 4.0]]]) # (2, 1, 2)
    y = torch.tensor([[[1.0]]])
    
    x_norm, _y_out = norm((x, y))
    
    # Chan 1: (12-10)/2 = 1.0, (8-10)/2 = -1.0
    # Chan 2: (6-5)/1 = 1.0, (4-5)/1 = -1.0
    expected_x = torch.tensor([[[1.0, -1.0]], [[1.0, -1.0]]])
    assert torch.allclose(x_norm, expected_x)
    assert torch.equal(_y_out, y)

def test_min_max_normalisation(tmp_path):
    stats_dir = tmp_path / "stats_minmax"
    stats_dir.mkdir()
    stats_file = stats_dir / "statistics.json"
    with stats_file.open("w") as f:
        json.dump({"chan1": {"min": 0.0, "max": 10.0}}, f)
    
    norm = MinMaxNormalisation(stats_dir, output_norm=True)
    x = torch.tensor([[[5.0]]])
    y = torch.tensor([[[5.0]]])
    
    x_norm, y_norm = norm((x, y))
    assert x_norm[0,0,0] == 0.5
    assert y_norm[0,0,0] == 0.5

def test_pad_unpad():
    fill_ref = -999.0
    pad = Pad(fill_value=fill_ref)
    unpad = UnPad(initial_size=(10, 10))
    
    # Input 10x10 -> Pad to 32x32
    x = [torch.ones((10, 10))]
    y = [torch.ones((10, 10))]
    
    x_pad, y_pad = pad((x, y))
    assert x_pad.shape == (1, 32, 32)
    assert y_pad.shape == (1, 32, 32)
    assert x_pad[0, 0, 0] == fill_ref # Check padding area
    
    y_unpad = unpad(y_pad)
    assert y_unpad.shape == (1, 10, 10)
    assert torch.all(y_unpad == 1.0)

def test_log10_transform():
    # Test with 'pr input' channel
    trans = Log10Transform(channels=['elevation', 'pr input'])
    x = torch.tensor([[[10.0]], [[9.0]]]) # chan 0=10, chan 1=9
    y = torch.tensor([[[1.0]]])
    
    x_out, _y_out = trans((x, y))
    
    # chan 1: log10(1 + 9) = 1.0
    assert torch.isclose(x_out[1, 0, 0], torch.tensor(1.0))
    # chan 0 should be unchanged
    assert x_out[0, 0, 0] == 10.0
