import os

import xarray as xr

# Prediction file parameters from verify_checkpoint.sh
PRED_FILE = "prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20100101_20101231_exp5_unet_gcm_bc.nc"

def main():
    if not os.path.exists(PRED_FILE):
        print(f"ERROR: Prediction file not found at {PRED_FILE}")
        return

    ds = xr.open_dataset(PRED_FILE)
    
    # Calculate global mean statistics for the predicted variable ('tas')
    if 'tas' not in ds:
        print(f"ERROR: Variable 'tas' not found in {PRED_FILE}")
        return

    mean_tas = ds.tas.mean().values
    std_tas = ds.tas.std().values
    min_tas = ds.tas.min().values
    max_tas = ds.tas.max().values

    print(f"--- DIAGNOSTIC BIAS CHECK: {PRED_FILE} ---")
    print(f"Global Mean Temperature: {mean_tas:.4f} K")
    print(f"Standard Deviation:      {std_tas:.4f} K")
    print(f"Min/Max Temperature:     {min_tas:.4f} K / {max_tas:.4f} K")

    # Quick consistency check against historical norm (~284 K)
    if 270 < mean_tas < 300:
        print("SUCCESS: Mean temperature is within the expected range (~284 K).")
    else:
        print("WARNING: Significant bias detected in mean temperature!")

if __name__ == "__main__":
    main()
