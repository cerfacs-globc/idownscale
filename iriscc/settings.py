"""
Experimental settings for all configurations.

date : 16/07/2025
author : Zoé GARCIA
"""

import os
from pathlib import Path

import pandas as pd

try:
    from iriscc import settings_local as _settings_local
except ImportError:  # pragma: no cover - optional local user configuration
    _settings_local = None

try:
    import cartopy.crs as ccrs
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    ccrs = None

try:
    import pyproj
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    pyproj = None


def local_setting(name: str, default=None):
    if _settings_local is None:
        return default
    paths = getattr(_settings_local, "PATHS", {})
    if name in paths:
        return paths[name]
    return getattr(_settings_local, name, default)


def env_path(name: str, default: Path | str) -> Path:
    configured = os.getenv(name)
    if configured is None:
        configured = local_setting(name, default)
    return Path(str(configured)).expanduser()


def _user_scratch_root() -> Path:
    user = os.getenv("USER") or os.getenv("LOGNAME") or "user"
    return Path("/scratch/globc") / user


def default_runtime_root() -> Path:
    if "IDOWNSCALE_RUNTIME_ROOT" in os.environ:
        return env_path("IDOWNSCALE_RUNTIME_ROOT", _user_scratch_root() / "idownscale_runtime")
    local_runtime = local_setting("IDOWNSCALE_RUNTIME_ROOT")
    if local_runtime is not None:
        return Path(str(local_runtime)).expanduser()
    return _user_scratch_root() / "idownscale_runtime"


def default_raw_dir() -> Path:
    if "IDOWNSCALE_RAW_DIR" in os.environ:
        return env_path("IDOWNSCALE_RAW_DIR", PROJECT_ROOT / "rawdata")
    local_raw = local_setting("IDOWNSCALE_RAW_DIR")
    if local_raw is not None:
        return Path(str(local_raw)).expanduser()
    repo_raw = PROJECT_ROOT / "rawdata"
    if repo_raw.exists():
        return repo_raw
    return RUNTIME_ROOT / "rawdata"


def default_output_dir() -> Path:
    if "IDOWNSCALE_OUTPUT_DIR" in os.environ:
        return env_path("IDOWNSCALE_OUTPUT_DIR", RUNTIME_ROOT / "output")
    local_output = local_setting("IDOWNSCALE_OUTPUT_DIR")
    if local_output is not None:
        return Path(str(local_output)).expanduser()
    return RUNTIME_ROOT / "output"


def safe_mkdir(directory: Path) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        # CI and other restricted environments may not be allowed to materialize
        # research/HPC default paths at import time.
        pass


def env_str(name: str, default: str) -> str:
    return os.getenv(name, str(local_setting(name, default)))


def plate_carree():
    return ccrs.PlateCarree() if ccrs is not None else None


def lambert_conformal(**kwargs):
    return ccrs.LambertConformal(**kwargs) if ccrs is not None else None


def pyproj_proj(spec: str, **kwargs):
    return pyproj.Proj(spec, **kwargs) if pyproj is not None else None

# Base directories
PROJECT_ROOT = Path(__file__).parents[1].resolve()
REPO_DIR = PROJECT_ROOT
RUNTIME_ROOT = default_runtime_root()
RAW_DIR = default_raw_dir()
OUTPUT_DIR = default_output_dir()

SAFRAN_DIR = RAW_DIR / "safran"
SAFRAN_RAW_DIR = SAFRAN_DIR / "raw_safran"
SAFRAN_REFORMAT_DIR = SAFRAN_DIR / "safran_reformat_day"
GCM_RAW_DIR = RAW_DIR / "gcm"
RCM_RAW_DIR = RAW_DIR / "rcm"
ERA5_DIR = RAW_DIR / "era5"
EOBS_RAW_DIR = RAW_DIR / "eobs"
CERRA_RAW_DIR = env_path("IDOWNSCALE_CERRA_DIR", RAW_DIR / "cerra")
CERRA_WORK_DIR = CERRA_RAW_DIR
ALADIN_RAW_DIR = RAW_DIR / "ALADIN"
TARGET_SAFRAN_FILE = SAFRAN_REFORMAT_DIR / "tas_day_SAFRAN_1959_reformat.nc"
TARGET_EOBS_EUROPE_FILE = EOBS_RAW_DIR / "tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc"
TARGET_EOBS_FRANCE_FILE = EOBS_RAW_DIR / "tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc"
TARGET_CERRA_FRANCE_FILE = CERRA_WORK_DIR / "tas_day_CERRA_19840901_20210910_france_reference.nc"
TARGET_GCM_FILE = GCM_RAW_DIR / "CNRM-CM6-1/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc"
OROG_EOBS_EUROPE_FILE = EOBS_RAW_DIR / "elevation_ens_025deg_reg_v29_0e.nc"
OROG_EOBS_FRANCE_FILE = EOBS_RAW_DIR / "elevation_ens_025deg_reg_v29_0e_france.nc"
OROG_CERRA_FRANCE_FILE = CERRA_WORK_DIR / "elevation_CERRA_france.nc"
OROG_SAFRAN_FILE = RAW_DIR / "topography/topography_safran2.nc"
IMERG_MASK = RAW_DIR / "landseamask/IMERG_land_sea_mask_regrid.nc" # only continents
LANDSEAMASK_GCM = GCM_RAW_DIR / "CNRM-CM6-1/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc"
LANDSEAMASK_ERA5 = ERA5_DIR / "lsm_ERA5.nc"
LANDSEAMASK_EOBS = EOBS_RAW_DIR / "eobs_landseamask.nc"
LANDSEAMASK_EOBS_FRANCE = EOBS_RAW_DIR / "eobs_landseamask_france.nc"
COUNTRIES_MASK = RAW_DIR /"landseamask/CNTR_RG_10M_2024_4326.nc"
ERA5_OROG_FILE = ERA5_DIR / "orography_ERA5.nc"
GCM_OROG_FILE = GCM_RAW_DIR / "orog_Emon_CNRM-CM6-1_historical_r10i1p1f2_gr_185001-201412.nc"
GCM_BC_DIR = env_path("IDOWNSCALE_GCM_BC_DIR", GCM_RAW_DIR / "CNRM-CM6-1-BC")
RCM_BC_DIR = env_path("IDOWNSCALE_RCM_BC_DIR", RCM_RAW_DIR / "ALADIN-BC")
EXP5_ARCHIVE_DATASET_DIR = env_path(
    "IDOWNSCALE_EXP5_ARCHIVE_DATASET_DIR",
    OUTPUT_DIR / "datasets" / "dataset_exp5_30y",
)
LEGACY_DATASET_ROOTS = [
    Path(path).expanduser()
    for path in os.getenv("IDOWNSCALE_LEGACY_DATASET_ROOTS", "").split(os.pathsep)
    if path
]
REGRID_WEIGHTS_DIR = env_path("IDOWNSCALE_REGRID_WEIGHTS_DIR", OUTPUT_DIR / "regrid_weights")

def build_custom_obs_source() -> dict:
    spec = {
        "kind": "observation",
        "root": env_path("IDOWNSCALE_CUSTOM_OBS_DIR", RAW_DIR / "custom_obs"),
        "geometry": env_str("IDOWNSCALE_CUSTOM_OBS_GEOMETRY", "eobs"),
        "data_type": env_str("IDOWNSCALE_CUSTOM_OBS_DATA_TYPE", "eobs"),
        "mask_type": os.getenv("IDOWNSCALE_CUSTOM_OBS_MASK_TYPE"),
    }
    yearly_pattern = os.getenv("IDOWNSCALE_CUSTOM_OBS_YEARLY_PATTERN")
    yearly_patterns = os.getenv("IDOWNSCALE_CUSTOM_OBS_YEARLY_PATTERNS")
    glob_pattern = os.getenv("IDOWNSCALE_CUSTOM_OBS_GLOB_PATTERN", "{var}*")
    if yearly_patterns:
        spec["yearly_patterns"] = [pattern.strip() for pattern in yearly_patterns.split(";") if pattern.strip()]
    elif yearly_pattern:
        spec["yearly_pattern"] = yearly_pattern
    else:
        spec["glob_pattern"] = glob_pattern
    return spec


# Source catalog
# Keep data-source wiring in one place so swapping reanalysis/model/target does
# not require edits spread across loaders and workflows.
SOURCE_CATALOG = {
    "era5": {
        "kind": "reanalysis",
        "root": ERA5_DIR,
        "geometry": "era5",
        "data_type": "era5",
        "output_label": "ERA5",
        "yearly_patterns": [
            "{var}_1d/{var}_1d_{year}_ERA5.nc",
            "{var}/{var}_1d_{year}_ERA5.nc",
        ],
        "orography_file": ERA5_OROG_FILE,
    },
    "era6": {
        "kind": "reanalysis",
        "root": env_path("IDOWNSCALE_ERA6_DIR", RAW_DIR / "era6"),
        "geometry": "era5",
        "data_type": "era6",
        "output_label": "ERA6",
        "yearly_patterns": [
            "{var}_1d/{var}_1d_{year}_ERA6.nc",
            "{var}/{var}_1d_{year}_ERA6.nc",
            "{var}*{year}*ERA6*.nc",
        ],
        "orography_file": env_path("IDOWNSCALE_ERA6_OROG_FILE", RAW_DIR / "era6/orography_ERA6.nc"),
    },
    "gcm_cnrm_cm6_1": {
        "kind": "model",
        "root": GCM_RAW_DIR / "CNRM-CM6-1",
        "geometry": "gcm",
        "data_type": "gcm",
        "output_label": "CNRM-CM6-1",
        "member_id": "r1i1p1f2",
        "grid_label": "gr",
        "historical_pattern": "{var}*historical*r1i1p1f2*",
        "scenario_pattern": "{var}*{ssp}*",
        "orography_file": GCM_OROG_FILE,
        "bias_corrected_root": GCM_BC_DIR,
    },
    "cordex": {
        "kind": "model",
        "root": env_path("IDOWNSCALE_CORDEX_DIR", RCM_RAW_DIR / "CORDEX"),
        "geometry": "rcm",
        "data_type": "rcm",
        "output_label": "CORDEX",
        "member_id": "r1i1p1f2",
        "grid_label": "gr",
        "historical_pattern": "{var}*historical*",
        "scenario_pattern": "{var}*{ssp}*",
        "orography_file": env_path("IDOWNSCALE_CORDEX_OROG_FILE", RCM_RAW_DIR / "CORDEX/orography_CORDEX.nc"),
        "bias_corrected_root": env_path("IDOWNSCALE_CORDEX_BC_DIR", RCM_RAW_DIR / "CORDEX-BC"),
    },
    "rcm_aladin": {
        "kind": "model",
        "root": RCM_RAW_DIR / "ALADIN",
        "geometry": "rcm",
        "data_type": "rcm",
        "output_label": "ALADIN",
        "member_id": "r1i1p1f2",
        "grid_label": "gr_150km",
        "historical_pattern": "{var}*historical*r1i1p1f2*",
        "scenario_pattern": "{var}*{ssp}*r1i1p1f2*",
        "bias_corrected_root": RCM_BC_DIR,
    },
    "eobs": {
        "kind": "observation",
        "root": EOBS_RAW_DIR,
        "geometry": "eobs",
        "data_type": "eobs",
        "glob_pattern": "{var}*",
        "mask_type": "eobs",
    },
    "cerra": {
        "kind": "observation",
        "root": CERRA_RAW_DIR,
        "geometry": "cerra",
        "data_type": "cerra",
        "yearly_pattern": "{var}_3h/{var}_3h_CERRA_{year}_*.nc",
        "combine_files_by": "time",
        "native_frequency": "3h",
        "default_frequency": "daily",
        "daily_aggregation": "mean",
        "mask_type": None,
    },
    "safran": {
        "kind": "observation",
        "root": SAFRAN_REFORMAT_DIR,
        "geometry": "safran",
        "data_type": "safran",
        "yearly_pattern": "{var}*{year}_reformat.nc",
    },
    "custom_obs": build_custom_obs_source(),
}


def get_source_spec(source_name: str) -> dict:
    if source_name not in SOURCE_CATALOG:
        raise KeyError(f"Unknown source '{source_name}'. Add it to SOURCE_CATALOG in iriscc/settings.py.")
    return SOURCE_CATALOG[source_name]


def get_source_output_label(source_name: str) -> str:
    return get_source_spec(source_name).get("output_label", source_name.upper())


def get_source_member_id(source_name: str) -> str:
    return get_source_spec(source_name).get("member_id", "r1i1p1f2")


def get_source_grid_label(source_name: str) -> str:
    return get_source_spec(source_name).get("grid_label", "gr")


def get_source_scenario_start(source_name: str) -> pd.Timestamp:
    env_override = {
        "gcm_cnrm_cm6_1": "IDOWNSCALE_GCM_SCENARIO_START",
        "cordex": "IDOWNSCALE_CORDEX_SCENARIO_START",
        "rcm_aladin": "IDOWNSCALE_RCM_SCENARIO_START",
    }.get(source_name)
    raw_value = os.getenv(env_override) if env_override is not None else None
    if raw_value is None:
        raw_value = get_source_spec(source_name).get("scenario_start")
    if raw_value is None:
        return pd.Timestamp(DATES_BC_TEST_FUTURE[0])
    return pd.Timestamp(raw_value)


def get_simu_source(exp: str, simu: str) -> str:
    if simu == "gcm":
        return CONFIG[exp].get("gcm_source", "gcm_cnrm_cm6_1")
    if simu == "rcm":
        return CONFIG[exp].get("rcm_source", "rcm_aladin")
    if simu in SOURCE_CATALOG:
        spec = get_source_spec(simu)
        if spec.get("kind") != "model":
            raise ValueError(
                f"Simulation source '{simu}' is not a model source. "
                "Use a model key from SOURCE_CATALOG or the 'gcm'/'rcm' aliases."
            )
        return simu
    raise ValueError(
        f"Unsupported simulation source '{simu}'. "
        "Use a model key from SOURCE_CATALOG or the 'gcm'/'rcm' aliases."
    )


def get_simu_family(exp: str, simu: str) -> str:
    resolved = get_simu_source(exp, simu)
    geometry = get_source_spec(resolved).get("geometry")
    if geometry == "gcm":
        return "gcm"
    if geometry == "rcm":
        return "rcm"
    raise ValueError(f"Model source '{resolved}' does not advertise a supported simulation geometry.")


def get_variant_source(exp: str, simu_variant: str) -> str:
    return get_simu_source(exp, simu_variant[:-3] if simu_variant.endswith("_bc") else simu_variant)


def get_bias_corrected_root(exp: str, simu: str) -> Path:
    root = get_source_spec(get_simu_source(exp, simu)).get("bias_corrected_root")
    if root is None:
        raise ValueError(f"No bias-corrected output directory configured for simulation family '{simu}'.")
    return Path(root)


def normalize_bc_tag(bc_tag: str | None) -> str:
    if not bc_tag:
        return ""
    return str(bc_tag).strip().replace(" ", "_")


def get_bc_bundle_path(exp: str, simu: str, period: str) -> Path:
    valid_periods = {"train_hist", "test_hist", "test_future"}
    if period not in valid_periods:
        raise ValueError(f"Unsupported BC bundle period '{period}'. Expected one of {sorted(valid_periods)}.")
    return DATASET_BC_DIR / f"bc_{period}_{exp}_{simu}.npz"


def get_bias_corrected_netcdf_path(
    exp: str,
    simu: str,
    var: str,
    period: str,
    ssp: str | None = None,
    bc_tag: str | None = None,
) -> Path:
    source_name = get_simu_source(exp, simu)
    scenario = "historical"
    if period == "train_hist":
        dates = get_bc_train_hist_dates(exp)
        date_range = f"{dates[0].strftime('%Y%m%d')}-{dates[-1].strftime('%Y%m%d')}"
    elif period == "test_hist":
        dates = get_bc_test_hist_dates(exp)
        date_range = f"{dates[0].strftime('%Y%m%d')}-{dates[-1].strftime('%Y%m%d')}"
    elif period == "test_future":
        scenario = ssp or CONFIG[exp].get("ssp", "ssp585")
        dates = get_bc_test_future_dates(exp)
        date_range = f"{dates[0].strftime('%Y%m%d')}-{dates[-1].strftime('%Y%m%d')}"
    else:
        raise ValueError(f"Unsupported BC period '{period}'.")
    tag_suffix = f"_{normalize_bc_tag(bc_tag)}" if normalize_bc_tag(bc_tag) else ""
    return get_bias_corrected_root(exp, simu) / (
        f"{var}_day_{get_source_output_label(source_name)}_{scenario}_"
        f"{get_source_member_id(source_name)}_{get_source_grid_label(source_name)}_{date_range}_bc{tag_suffix}.nc"
    )


def get_bias_corrected_sample_dir(exp: str, simu: str, bc_tag: str | None = None) -> Path:
    tag_suffix = f"_{normalize_bc_tag(bc_tag)}" if normalize_bc_tag(bc_tag) else ""
    return DATASET_BC_DIR / f"dataset_{exp}_test_{simu}_bc{tag_suffix}"


def get_prediction_output_path(
    exp: str,
    simu_variant: str,
    var: str,
    startdate: str,
    enddate: str,
    test_name: str,
    ssp: str | None = None,
) -> Path:
    source_name = get_variant_source(exp, simu_variant)
    scenario_start = get_source_scenario_start(source_name)
    period = "historical" if pd.Timestamp(enddate) < scenario_start else (ssp or CONFIG[exp].get("ssp", "ssp585"))
    return PREDICTION_DIR / (
        f"{var}_day_{get_source_output_label(source_name)}_{period}_"
        f"{get_source_member_id(source_name)}_{get_source_grid_label(source_name)}_"
        f"{startdate}_{enddate}_{exp}_{test_name}.nc"
    )


def get_metrics_test_name(test_name: str, simu_test: str | None = None) -> str:
    if simu_test:
        return f"{test_name}_{simu_test}"
    return test_name


def get_value_metrics_output_path(exp: str, test_name: str, simu_test: str | None = None) -> Path:
    metrics_test_name = get_metrics_test_name(test_name, simu_test)
    return METRICS_DIR / exp / f"value_metrics_{exp}_{metrics_test_name}.csv"


def get_dataset_variant_dir(exp: str, variant: str) -> Path:
    return DATASET_BC_DIR / f"dataset_{exp}_test_{variant}"


def get_evaluation_sample_dir(exp: str, test_name: str, simu_test: str | None = None) -> Path | None:
    if CONFIG.get(exp, {}).get("target") == "perfect_model":
        configured_eval = CONFIG.get(exp, {}).get("evaluation_dataset")
        if configured_eval is not None:
            return Path(configured_eval)
        if simu_test:
            return get_dataset_variant_dir(exp, simu_test)
        return Path(CONFIG[exp]["dataset"])
    if test_name.startswith("baseline"):
        return DATASET_DIR / f"dataset_{exp}_baseline"
    if test_name == "era5_raw":
        return DATASET_DIR / f"dataset_{exp}_30y"
    if test_name.endswith("_raw"):
        return get_dataset_variant_dir(exp, test_name[:-4])
    if simu_test:
        return get_dataset_variant_dir(exp, simu_test)
    return None


def get_input_channel_index(exp: str, var: str | None = None) -> int:
    cfg = CONFIG[exp]
    if "raw_input_channel_index" in cfg:
        return int(cfg["raw_input_channel_index"])
    input_vars = cfg.get("input_vars", [])
    if var is not None and var in input_vars:
        return int(input_vars.index(var))
    return max(len(input_vars) - 1, 0)


def get_bc_input_channel_index(exp: str) -> int | None:
    value = CONFIG[exp].get("bc_input_channel_index")
    return None if value is None else int(value)


def get_target_channel_index(exp: str, var: str | None = None) -> int:
    target_vars = CONFIG[exp].get("target_vars", [])
    if var is not None and var in target_vars:
        return int(target_vars.index(var))
    return 0

# Results redirected to OUTPUT_DIR
DATASET_DIR = env_path("IDOWNSCALE_DATASET_DIR", OUTPUT_DIR / "datasets")
DATASET_EXP1_DIR = DATASET_DIR / "dataset_exp1"
DATASET_EXP1_CONTINENTS_DIR = DATASET_DIR / "dataset_exp1_continents"
DATASET_EXP1_30Y_DIR = DATASET_DIR / "dataset_exp1_30y"
DATASET_EXP1_6MB_DIR = DATASET_DIR / "dataset_exp1_6mb"
DATASET_EXP1_6MB_30Y_DIR = DATASET_DIR / "dataset_exp1_6mb_30y"
DATASET_EXP2_DIR = DATASET_DIR / "dataset_exp2"
DATASET_EXP2_6MB_DIR = DATASET_DIR / "dataset_exp2_6mb"
DATASET_EXP2_BI_DIR = DATASET_DIR / "dataset_exp2_bi"
DATASET_EXP3_30Y_DIR = DATASET_DIR / "dataset_exp3_30y"
DATASET_EXP3_BASELINE_DIR = DATASET_DIR / "dataset_exp3_baseline"
DATASET_EXP4_30Y_DIR = DATASET_DIR / "dataset_exp4_30y"
DATASET_EXP4_BASELINE_DIR = DATASET_DIR / "dataset_exp4_baseline"
DATASET_EXP5_30Y_DIR = DATASET_DIR / "dataset_exp5_30y"
DATASET_EXPC_30Y_DIR = DATASET_DIR / "dataset_expc_37y"
DATASET_EXP6_30Y_DIR = DATASET_DIR / "dataset_exp6_30y"
DATASET_EXP6_BASELINE_DIR = DATASET_DIR / "dataset_exp6_baseline"
DATASET_EXP7_30Y_DIR = DATASET_DIR / "dataset_exp7_30y"
DATASET_EXP8_30Y_DIR = DATASET_DIR / "dataset_exp8_30y"
DATASET_TEST_ERA5_DIR = DATASET_DIR / "dataset_test_era5"
DATASET_BC_DIR = env_path("IDOWNSCALE_DATASET_BC_DIR", DATASET_DIR / "dataset_bc")

RUNS_DIR = env_path("IDOWNSCALE_RUNS_DIR", OUTPUT_DIR / "runs")
GRAPHS_DIR = env_path("IDOWNSCALE_GRAPHS_DIR", RUNTIME_ROOT / "graphs")
METRICS_DIR = env_path("IDOWNSCALE_METRICS_DIR", OUTPUT_DIR / "metrics")
PREDICTION_DIR = env_path("IDOWNSCALE_PREDICTION_DIR", OUTPUT_DIR / "prediction")

for directory in [
    OUTPUT_DIR,
    REGRID_WEIGHTS_DIR,
    DATASET_DIR,
    DATASET_BC_DIR,
    RUNS_DIR,
    GRAPHS_DIR,
    METRICS_DIR,
    PREDICTION_DIR,
    GCM_BC_DIR,
    RCM_BC_DIR,
] + [
    Path(spec["bias_corrected_root"])
    for spec in SOURCE_CATALOG.values()
    if spec.get("bias_corrected_root") is not None
]:
    safe_mkdir(Path(directory))

CONFIG = {
    "exp3":
        {"target":"safran",
            "domain": [-6., 12., 40., 52.],
            "domain_xy" : [60000, 1196000, 1617000, 2681000],
            "data_projection" : lambert_conformal(central_longitude=2.337229,
                                    central_latitude=46.8,
                                    false_easting=600000,
                                    false_northing=2200000,
                                    standard_parallels=(45.89892, 47.69601)),
            "fig_projection" : lambert_conformal(central_latitude=46., central_longitude=2.),
            "pyproj_projection" : pyproj_proj("+proj=lcc +lon_0=2.337229 +lat_0=46.8 +lat_1=45.89892 +lat_2=47.69601 +x_0=600000 +y_0=2200000"),
            "shape" : (134,143),
            "target_file" : OROG_SAFRAN_FILE,
            "orog_file" : OROG_SAFRAN_FILE,
            "dataset" : DATASET_EXP3_30Y_DIR,
            "target_vars": ["tas"],
            "input_vars": ["elevation", "tas"],
            "channels": ["elevation", "tas input", "tas target"]
            },

    "exp4": # obsolete, use exp5
        {"target":"eobs",
            "domain":
                {"france": [-6., 10., 38, 54],
                "europe" : [-12.5, 27.5, 31., 71.],
                "tchequie" : [11.5, 19.5, 45.75, 53.75]},
            "data_projection" : plate_carree(),
            "fig_projection" :
                {"france" : lambert_conformal(central_latitude=46., central_longitude=2.),
                "europe" : lambert_conformal(central_latitude=51., central_longitude=7.5),
                "tchequie" : lambert_conformal(central_latitude=45.75, central_longitude=11.5)},
            "shape":
                    {"france": (64,64),
                    "europe" : (160,160),
                    "tchequie" : (32,32)},
            "target_file" : EOBS_RAW_DIR / "tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc",
            "orog_file" : EOBS_RAW_DIR / "elevation_ens_025deg_reg_v29_0e_france.nc",
            "dataset" : DATASET_EXP4_30Y_DIR,
            "target_vars": ["tas"],
            "input_vars": ["elevation", "tas"],
            "channels": ["elevation", "tas input", "tas target"]
            },
    "exp5":
        {"target":"eobs",
            "domain": [-6.0, 10.0, 38.0, 54.0],
            "bc_domain": [-12.5, 27.5, 31.0, 71.0],
            "bias_correction_method": "ibicus_cdft",
            "phase1_reanalysis_source": "era5",
            "bc_reanalysis_source": "era5",
            "gcm_source": "gcm_cnrm_cm6_1",
            "rcm_source": "rcm_aladin",
            "target_source": "eobs",
            "data_projection" : plate_carree(),
            "fig_projection" : lambert_conformal(central_latitude=46., central_longitude=2.),
            "pyproj_projection" : None,
            "shape": (64,64),
            "target_file" : TARGET_EOBS_FRANCE_FILE,
            "orog_file" : OROG_EOBS_FRANCE_FILE,
            "dataset" : DATASET_EXP5_30Y_DIR,
            "target_vars": ["tas"],
            "input_vars": ["elevation", "tas"],
            "channels": ["elevation", "tas input", "tas target"],
            "ssp": "ssp585",
            "model": "unet",
            "lapse_rate_correction": False,
            "fill_value": 0.0,
            "target_source_pregridded": False,
            "phase1_bridge_method": "conservative_normed",
            "phase1_target_method": "conservative_normed",
            "phase1_crop_target": True,
            "perfect_model_input_coarse_method": "conservative_normed",
            "perfect_model_input_target_method": "bilinear",
            "perfect_model_target_method": "conservative_normed",
            "phase1_start_date": "1980-01-01",
            "phase1_end_date": "2014-12-31",
            "train_split_dates": ["19800101", "20100101", "20140101"],
            "bc_train_hist_start_date": "1980-01-01",
            "bc_train_hist_end_date": "1999-12-31",
            "bc_test_hist_start_date": "2000-01-01",
            "bc_test_hist_end_date": "2014-12-31",
            "bc_test_future_start_date": "2015-01-01",
            "bc_test_future_end_date": "2100-12-31",
            },
    "expc":
        {"target":"cerra",
            "domain": [-6.0, 10.0, 38.0, 54.0],
            "bc_domain": [-12.5, 27.5, 31.0, 71.0],
            "bias_correction_method": "ibicus_cdft",
            "phase1_reanalysis_source": "era5",
            "bc_reanalysis_source": "era5",
            "gcm_source": "gcm_cnrm_cm6_1",
            "rcm_source": "rcm_aladin",
            "target_source": "cerra",
            "data_projection" : plate_carree(),
            "fig_projection" : lambert_conformal(central_latitude=46., central_longitude=2.),
            "pyproj_projection" : None,
            "shape": (503,326),
            "target_file" : TARGET_CERRA_FRANCE_FILE,
            "orog_file" : OROG_CERRA_FRANCE_FILE,
            "dataset" : DATASET_EXPC_30Y_DIR,
            "target_vars": ["tas"],
            "input_vars": ["elevation", "tas"],
            "channels": ["elevation", "tas input", "tas target"],
            "ssp": "ssp585",
            "model": "unet",
            "lapse_rate_correction": False,
            "fill_value": 0.0,
            "target_source_pregridded": False,
            "phase1_bridge_method": "conservative_normed",
            "phase1_target_method": "bilinear",
            "phase1_crop_target": False,
            "perfect_model_input_coarse_method": "conservative_normed",
            "perfect_model_input_target_method": "bilinear",
            "perfect_model_target_method": "bilinear",
            "phase1_start_date": "1984-09-01",
            "phase1_end_date": "2021-09-10",
            "train_split_dates": ["19840901", "20170101", "20200101"],
            "bc_train_hist_start_date": "1984-09-01",
            "bc_train_hist_end_date": "1999-12-31",
            "bc_test_hist_start_date": "2000-01-01",
            "bc_test_hist_end_date": "2021-09-10",
            "bc_test_future_start_date": "2021-09-11",
            "bc_test_future_end_date": "2100-12-31",
            },
    "exp6":
        {"target":"eobs",
            "domain": [-6., 10., 38, 54],
            "bias_correction_method": "ibicus_cdft",
            "phase1_reanalysis_source": "era5",
            "bc_reanalysis_source": "era5",
            "gcm_source": "gcm_cnrm_cm6_1",
            "rcm_source": "rcm_aladin",
            "target_source": "eobs",
            "data_projection" : plate_carree(),
            "fig_projection" : lambert_conformal(central_latitude=46., central_longitude=2.),
            "pyproj_projection" : None, # for curvilign grids conservative interpolation
            "shape": (64,64),
            "target_file" : TARGET_EOBS_FRANCE_FILE, # target grid coordinates
            "orog_file" : OROG_EOBS_FRANCE_FILE,
            "dataset" : DATASET_EXP6_30Y_DIR,
            "target_vars": ["pr"],
            "input_vars": ["elevation", "pr"],
            "channels": ["elevation", "pr input", "pr target"], # to not get lost for normalization
            "ssp" : "ssp585"
            },
    "exp7":
        {"target":"eobs",
            "domain": [-6., 10., 38, 54],
            "bias_correction_method": "ibicus_cdft",
            "phase1_reanalysis_source": "era5",
            "bc_reanalysis_source": "era5",
            "gcm_source": "gcm_cnrm_cm6_1",
            "rcm_source": "rcm_aladin",
            "target_source": "eobs",
            "data_projection" : plate_carree(),
            "fig_projection" : lambert_conformal(central_latitude=46., central_longitude=2.),
            "pyproj_projection" : None, # for curvilign grids conservative interpolation
            "shape": (64,64),
            "target_file" : TARGET_EOBS_FRANCE_FILE, # target grid coordinates
            "orog_file" : OROG_EOBS_FRANCE_FILE,
            "dataset" : DATASET_EXP7_30Y_DIR,
            "target_vars": ["pr"],
            "input_vars": ["elevation", "huss", "psl", "tas"],
            "channels": ["elevation", "huss input", "psl input", "tas input", "pr target"], # to not get lost for normalization
            "ssp" : "ssp585"
            },
    "exp8":
        {"target":"eobs",
            "domain": [-6., 10., 38, 54],
            "bias_correction_method": "ibicus_cdft",
            "phase1_reanalysis_source": "era5",
            "bc_reanalysis_source": "era5",
            "gcm_source": "gcm_cnrm_cm6_1",
            "rcm_source": "rcm_aladin",
            "target_source": "eobs",
            "data_projection" : plate_carree(),
            "fig_projection" : lambert_conformal(central_latitude=46., central_longitude=2.),
            "pyproj_projection" : None, # for curvilign grids conservative interpolation
            "shape": (64,64),
            "target_file" : TARGET_EOBS_FRANCE_FILE, # target grid coordinates
            "orog_file" : OROG_EOBS_FRANCE_FILE,
            "dataset" : DATASET_EXP8_30Y_DIR,
            "target_vars": ["pr"],
            "input_vars": ["elevation",
                        "zg500",
                        "zg700",
                        "zg850",
                        "ta500",
                        "ta700",
                        "ta850",
                        "ua500",
                        "ua700",
                        "ua850",
                        "vas",
                        "uas",
                        "psl"],
            "channels": ["elevation",
                        "zg500 input",
                        "zg700 input",
                        "zg850 input",
                        "ta500 input",
                        "ta700 input",
                        "ta850 input",
                        "ua500 input",
                        "ua700 input",
                        "ua850 input",
                        "vas input",
                        "uas input",
                        "psl input",
                        "pr target"], # to not get lost for normalization
            "ssp" : "ssp585"
            }
    }

COLORS = {"SAFRAN 8km": "purple",
          "E-OBS 25km" : "blue",
          "ERA5 8km" : "cyan",
          "ERA5 0.25°" : "green",
          "GCM 1°" : "orange",
          "RCM 12km" : "orange",
          "UNet" : "orangered",
          "SwinUNETR" : "hotpink"}

ALADIN_PROJ_PYPROJ = pyproj_proj(
    "+proj=lcc +lat_1=49.500000 +lat_0=49.500000 +lon_0=10.500000 +k_0=1.0 +x_0=2925000.000000 +y_0=2925000.000000 +R=6371229.000000",
    preserve_units=True,
)
SAFRAN_PROJ_PYPROJ = pyproj_proj(
    "+proj=lcc +lon_0=2.337229 +lat_0=46.8 +lat_1=45.89892 +lat_2=47.69601 +x_0=600000 +y_0=2200000"
)


# Phase 1 settings
#DATES = pd.date_range(start='19850101', end='2004-12-31', freq='D')
#DATES_TRAIN = ['1985', '2001', '2003'] # train, valid, test start (ex8 mini dataset fior test)
#DATES_TRAIN = ['1985', '2004', '2010'] # train, valid, test start
DATES_TEST = pd.date_range(start="2010-01-01", end="2014-12-31", freq="D")

# Phase 2 settings
DATES = pd.date_range(start="1980-01-01", end="2014-12-31", freq="D") # all data for phase 2
DATES_TRAIN = ["1980", "2010", "2014"] # train, valid, test start

DATES_BC_TRAIN_HIST = pd.date_range(start="1980-01-01", end="1999-12-31", freq="D")
DATES_BC_TEST_HIST = pd.date_range(start="2000-01-01", end="2014-12-31", freq="D")
DATES_BC_TEST_FUTURE = pd.date_range(start="2015-01-01", end="2100-12-31", freq="D")


def get_phase1_dates(exp: str) -> pd.DatetimeIndex:
    cfg = CONFIG[exp]
    return pd.date_range(start=cfg["phase1_start_date"], end=cfg["phase1_end_date"], freq="D")


def get_train_split_dates(exp: str) -> list[str]:
    cfg = CONFIG[exp]
    return list(cfg.get("train_split_dates", [f"{year}0101" for year in DATES_TRAIN]))


def get_bc_train_hist_dates(exp: str) -> pd.DatetimeIndex:
    cfg = CONFIG[exp]
    return pd.date_range(start=cfg["bc_train_hist_start_date"], end=cfg["bc_train_hist_end_date"], freq="D")


def get_bc_test_hist_dates(exp: str) -> pd.DatetimeIndex:
    cfg = CONFIG[exp]
    return pd.date_range(start=cfg["bc_test_hist_start_date"], end=cfg["bc_test_hist_end_date"], freq="D")


def get_bc_test_future_dates(exp: str) -> pd.DatetimeIndex:
    cfg = CONFIG[exp]
    return pd.date_range(start=cfg["bc_test_future_start_date"], end=cfg["bc_test_future_end_date"], freq="D")

CONFIG["exp5_audit"] = {
    "target":"eobs",
    "domain": [-12.5, 27.5, 31.0, 71.0],
    "bias_correction_method": "ibicus_cdft",
    "phase1_reanalysis_source": "era5",
    "bc_reanalysis_source": "era5",
    "gcm_source": "gcm_cnrm_cm6_1",
    "rcm_source": "rcm_aladin",
    "target_source": "eobs",
    "data_projection" : plate_carree(),
    "fig_projection" : lambert_conformal(central_latitude=51., central_longitude=7.5),
    "pyproj_projection" : None,
    "shape": (29, 28),
    "target_file" : EOBS_RAW_DIR / "tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc",
    "orog_file" : OROG_EOBS_FRANCE_FILE, # elevation is for reference, build_dataset_bc doesn't use it for BC
    "dataset" : DATASET_DIR / "dataset_exp5_30y",
    "target_vars": ["tas"],
    "input_vars": ["elevation", "tas"],
    "channels": ["elevation", "tas input", "tas target"],
    "ssp": "ssp585",
    "lapse_rate_correction": False
}

CONFIG["perfect_model_rcm"] = {
    **CONFIG["exp5"],
    "target": "perfect_model",
    "bc_reanalysis_source": "rcm_aladin",
    "target_source": "rcm_aladin",
    "dataset": DATASET_BC_DIR / "dataset_perfect_model_rcm_bcml_audit",
    "evaluation_dataset": DATASET_BC_DIR / "dataset_perfect_model_rcm_eval_bcml_audit",
    "perfect_model_input_source": "rcm_aladin",
    "perfect_model_condition_on_bc": True,
    "perfect_model_conditioning_bc_tag": None,
    "perfect_model_input_resolution": "150km",
    "perfect_model_input_grid_source": "gcm_cnrm_cm6_1",
    "perfect_model_input_coarse_method": "conservative_normed",
    "perfect_model_input_target_method": "bilinear",
    "perfect_model_target_source": "rcm_aladin",
    "perfect_model_target_resolution": "native",
    "perfect_model_target_method": "conservative_normed",
    "input_vars": ["elevation", "tas_coarse", "tas_bc"],
    "channels": ["elevation", "degraded model input", "bias-corrected model input", "native model target"],
    "raw_input_channel_index": 1,
    "bc_input_channel_index": 2,
}
