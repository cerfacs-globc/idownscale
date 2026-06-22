import pytest
import xarray as xr

from iriscc.datautils import Data, apply_landseamask


def test_resolve_source_file_raises_when_non_rcm_match_is_ambiguous(monkeypatch):
    data = Data("france")
    monkeypatch.setattr(data, "_resolve_source_files", lambda source_name, var, date=None, ssp=None: ["a.nc", "b.nc"])
    monkeypatch.setattr(data, "get_source_spec", lambda source_name: {"geometry": "gcm"})

    with pytest.raises(FileExistsError, match="Expected exactly one source file"):
        data._resolve_source_file("gcm_cnrm_cm6_1", "tas")


def test_resolve_source_file_prefers_france_eobs_target_when_available(monkeypatch):
    data = Data("france")
    monkeypatch.setattr(
        data,
        "_resolve_source_files",
        lambda source_name, var, date=None, ssp=None: [
            "/tmp/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc",
            "/tmp/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc",
        ],
    )
    monkeypatch.setattr(data, "get_source_spec", lambda source_name: {"geometry": "eobs"})

    resolved = data._resolve_source_file("eobs", "tas")

    assert resolved.endswith("_france.nc")


def test_apply_landseamask_uses_cropped_eobs_mask(monkeypatch):
    ds = xr.Dataset(
        {"tas": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"lat": [45.0, 46.0], "lon": [3.0, 4.0]},
    )
    full_mask = xr.Dataset(
        {
            "landseamask": (
                ("lat", "lon"),
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            )
        },
        coords={"lat": [44.0, 45.0, 46.0], "lon": [2.0, 3.0, 4.0]},
    )

    monkeypatch.setattr("iriscc.datautils.LANDSEAMASK_EOBS", "/tmp/fake_mask.nc")
    monkeypatch.setattr("iriscc.datautils.standardize_eobs_geometry", lambda dataset: dataset)
    monkeypatch.setattr("iriscc.datautils.xr.open_dataset", lambda *args, **kwargs: full_mask)

    masked = apply_landseamask(ds, "eobs", variables=["tas"])

    assert float(masked["tas"].sel(lat=45.0, lon=3.0)) != float(masked["tas"].sel(lat=45.0, lon=3.0))
    assert float(masked["tas"].sel(lat=45.0, lon=4.0)) == 2.0
