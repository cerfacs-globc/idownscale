from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import xarray as xr

from iriscc.windutils import direction_from_components, speed_from_components


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "bin/postprocessing/derive_wind_products.py"
    spec = spec_from_file_location("derive_wind_products", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_speed_from_components():
    np.testing.assert_allclose(speed_from_components(np.array([3.0]), np.array([4.0])), np.array([5.0]))


def test_direction_from_components_uses_meteorological_convention():
    np.testing.assert_allclose(direction_from_components(np.array([1.0]), np.array([0.0])), np.array([270.0]))
    np.testing.assert_allclose(direction_from_components(np.array([0.0]), np.array([1.0])), np.array([180.0]))
    np.testing.assert_allclose(direction_from_components(np.array([-1.0]), np.array([0.0])), np.array([90.0]))
    np.testing.assert_allclose(direction_from_components(np.array([0.0]), np.array([-1.0])), np.array([0.0]))


def test_validate_component_alignment_detects_coordinate_mismatch():
    module = load_module()
    ds_u = xr.Dataset({"uas": (("time", "x"), np.ones((1, 2)))}, coords={"time": [0], "x": [0, 1]})
    ds_v = xr.Dataset({"vas": (("time", "x"), np.ones((1, 2)))}, coords={"time": [0], "x": [0, 2]})

    try:
        module.validate_component_alignment(ds_u, ds_v, "uas", "vas")
    except ValueError as exc:
        assert "Coordinate 'x' values differ" in str(exc)
    else:
        raise AssertionError("Expected coordinate mismatch to raise ValueError")
