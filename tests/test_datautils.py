import numpy as np
import xarray as xr
import pytest

pytest.importorskip("esmpy")

from iriscc.datautils import (
    standardize_longitudes,
    generate_bounds,
)

def test_standardize_longitudes():
    # Test 1D lon
    ds = xr.Dataset(coords={'lon': [350.0, 10.0]})
    ds_std = standardize_longitudes(ds)
    assert np.all(ds_std.lon.values <= 180.0)
    assert np.all(ds_std.lon.values >= -180.0)
    assert -10.0 in ds_std.lon.values

    # Test 2D lon (mesh)
    ds2 = xr.Dataset(coords={'lon': (('y','x'), [[350.0]]), 'lat': (('y','x'), [[45.0]])})
    ds2_std = standardize_longitudes(ds2)
    assert ds2_std.lon.values[0,0] == -10.0

def test_generate_bounds():
    coord = np.array([10.0, 20.0, 30.0])
    bounds = generate_bounds(coord)
    # Expected: [5, 15, 25, 35]
    assert len(bounds) == 4
    assert bounds[1] == 15
    assert bounds[0] == 5
    assert bounds[3] == 35

from iriscc.datautils import Data
from iriscc.settings import DATASET_METADATA

def test_data_class_metadata_lookup():
    # Test if metadata is correctly loaded for a known source
    assert 'era5' in DATASET_METADATA
    assert DATASET_METADATA['era5']['var_map']['tas'] == 't2m'

def test_data_init():
    domain = [-6., 10., 38, 54]
    data = Data(domain)
    assert data.domain == domain
