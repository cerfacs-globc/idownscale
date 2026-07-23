from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import xarray as xr


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "bin/preprocessing/bias_correction_sbck.py"
    spec = spec_from_file_location("bias_correction_sbck", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_corrected_dataset_preserves_scalar_variable_name():
    module = load_module()
    reference = xr.Dataset(
        data_vars={"psl": (("lat", "lon"), np.array([[101000.0, 101100.0], [101200.0, 101300.0]]))},
        coords={"lat": [45.0, 46.0], "lon": [3.0, 4.0]},
    )
    values = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    dates = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")

    corrected = module.build_corrected_dataset(reference, "psl", values, dates)

    assert list(corrected.data_vars) == ["psl"]
    assert corrected["psl"].dims == ("time", "lat", "lon")
    np.testing.assert_array_equal(corrected["psl"].values, values)


def test_materialize_corrected_samples_uses_requested_scalar_variable(tmp_path, monkeypatch):
    module = load_module()
    corrected_ds = xr.Dataset(
        data_vars={"psl": (("time", "lat", "lon"), np.array([[[101000.0, 101100.0], [101200.0, 101300.0]]]))},
        coords={
            "time": np.array(["2000-01-01"], dtype="datetime64[ns]"),
            "lat": [45.0, 46.0],
            "lon": [3.0, 4.0],
        },
    )
    monkeypatch.setitem(module.CONFIG, "unit_test_exp", {"perfect_model_target_method": "conservative_normed"})
    monkeypatch.setattr(module, "reformat_as_target", lambda ds_day, **kwargs: ds_day)
    monkeypatch.setattr(
        module,
        "target_sample_for_date",
        lambda **kwargs: np.expand_dims(np.full((2, 2), 7.0, dtype=np.float32), axis=0),
    )

    module.materialize_corrected_samples(
        corrected_ds=corrected_ds,
        dates=corrected_ds.time.values,
        dataset_bc_dir=tmp_path,
        orog=np.full((2, 2), 42.0, dtype=np.float32),
        target_file=tmp_path / "target.nc",
        domain=[0.0, 1.0, 0.0, 1.0],
        get_data=object(),
        exp="unit_test_exp",
        var="psl",
        ssp="ssp585",
        include_target=True,
    )

    sample = np.load(tmp_path / "sample_20000101.npz")
    assert sample["x"].shape == (2, 2, 2)
    np.testing.assert_array_equal(sample["x"][0], np.full((2, 2), 42.0, dtype=np.float32))
    np.testing.assert_array_equal(sample["x"][1], corrected_ds["psl"].isel(time=0).values.astype(np.float32))
    np.testing.assert_array_equal(sample["y"], np.expand_dims(np.full((2, 2), 7.0, dtype=np.float32), axis=0))


def test_bc_dataset_array_time_validation_identifies_source_file():
    module_path = Path(__file__).resolve().parents[1] / "bin/preprocessing/build_dataset_bc.py"
    spec = spec_from_file_location("build_dataset_bc", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with np.testing.assert_raises_regex(
        ValueError,
        "source 'cerra'.*variable 'uas'.*uas_3h_CERRA_1985.nc.*has 1 time steps.*7305 were requested",
    ):
        module.require_expected_array_time_length(
            np.zeros((1, 44, 49), dtype=np.float32),
            np.arange("1985-01-01", "2005-01-01", dtype="datetime64[D]"),
            "cerra",
            "uas",
            "train_hist reference after regridding",
            "uas_3h_CERRA_1985.nc",
        )
