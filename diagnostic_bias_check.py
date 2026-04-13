import os
import glob
import xarray as xr
import numpy as np
from pathlib import Path

# Prediction file parameters from verify_checkpoint.sh
PRED_FILE = "prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20100101_20101231_exp5_unet_gcm_bc.nc"

def check_bias(file_path):
    if not os.path.exists(file_path):
        print(f"ERROR: Prediction file not found at {file_path}")
        return False
    
    print(f"--- DIAGNOSTIC BIAS CHECK: {file_path} ---")
    ds = xr.open_dataset(file_path)
    mean_temp = float(ds.tas.mean())
    std_temp = float(ds.tas.std())
    min_temp = float(ds.tas.min())
    max_temp = float(ds.tas.max())
    
    print(f"Global Mean Temperature: {mean_temp:.4f} K")
    print(f"Standard Deviation:      {std_temp:.4f} K")
    print(f"Min/Max Temperature:     {min_temp:.4f} K / {max_temp:.4f} K")
    
    # Scientific thresholds
    # Normal temp around 284 K.
    # Cold bias (26 K shift) would result in ~258 K.
    if mean_temp < 270:
        print("!!! CRITICAL WARNING: COLD BIAS DETECTED (Mean < 270 K) !!!")
        print("This suggests a missing Kelvin offset (273.15) or a denormalization failure.")
        return False
    elif mean_temp > 300:
        print("!!! WARNING: UNUSUALLY HIGH TEMPERATURE DETECTED !!!")
        return False
    else:
        print("SUCCESS: Mean temperature is within the expected range (~284 K).")
        return True

if __name__ == "__main__":
    success = check_bias(PRED_FILE)
    if not success:
        exit(1)
    exit(0)
