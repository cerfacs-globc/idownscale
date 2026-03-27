import numpy as np
import xarray as xr
import pytest

pytest.importorskip("esmpy")

from iriscc.datautils import (
    standardize_longitudes,
    generate_bounds,
    standardize_dims_and_coords,
    crop_domain_from_ds,
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

def test_standardize_dims_and_coords():
    ds = xr.Dataset(coords={'nav_lon': [0, 1], 'nav_lat': [0, 1]}, data_vars={'tas': (('nav_lat', 'nav_lon'), [[1, 2], [3, 4]])})
    ds = ds.rename({'nav_lon': 'nlon', 'nav_lat': 'nlat'}) # Mock some odd names
    ds_std = standardize_dims_and_coords(ds)
    assert 'lon' in ds_std.coords
    assert 'lat' in ds_std.coords

def test_crop_domain_from_ds():
    ds = xr.Dataset(coords={'lon': [0, 10, 20], 'lat': [0, 10, 20]}, data_vars={'tas': (('lat', 'lon'), np.zeros((3,3)))})
    domain = (5, 15, 5, 15)
    ds_cropped = crop_domain_from_ds(ds, domain)
    assert len(ds_cropped.lon) == 1
    assert ds_cropped.lon.values[0] == 10
