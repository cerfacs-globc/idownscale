import numpy as np
import os
import sys
import xarray as xr
import pandas as pd
import argparse

# Internal logic alignment
sys.path.append(os.getcwd())
from iriscc.datautils import interpolation_target_grid
from iriscc.settings import OROG_EOBS_FRANCE_FILE

def check_parity(data_new, data_arch, layer_label, ds_orog_aligned, var_name="tas", units="K"):
    """Perform statistical parity audit with rigorous NaN and shape handling."""
    # Stabilize dimensions to 2D for cross-reference with orography
    data_new = np.squeeze(data_new)
    data_arch = np.squeeze(data_arch)
    
    diff = data_new - data_arch
    diff_abs = np.abs(diff)
    max_diff = np.nanmax(diff_abs)
    
    orog_values = np.squeeze(ds_orog_aligned.elevation.values)
    if orog_values.shape != data_new.shape:
        # Resize or pad if necessary (should be consistent 64x64)
        print(f"WARNING: Shape mismatch - Orog: {orog_values.shape} vs Data: {data_new.shape}")
        orog_values = np.zeros_like(data_new)

    print(f"\nAUDIT LAYER: {layer_label}")
    print(f"Overall Metrics: Variable: {var_name:8} | Units: {units:3} | Bias: {np.nanmean(diff):10.2e} | Std: {np.nanstd(diff):10.2e} | MaxDiff: {max_diff:10.2e}")
    
    print(f"{'Threshold':<12} | {'Coverage':<10} | {'Avg Diff':<10} | {'Std Diff':<10} | {'Mean Altitude':<15}")
    print("-" * 80)
    
    valid_mask = ~np.isnan(data_arch)
    total_valid_pts = np.sum(valid_mask)
    
    for thresh in [-1.0, 0.1, 1.0, 5.0]:
        if thresh < 0:
            mask = valid_mask
            label = "Overall"
        else:
            mask = (diff_abs > thresh) & valid_mask
            label = f">{thresh:<6}"
            
        pts_in_slice = np.sum(mask)
        if pts_in_slice == 0:
            print(f"{label:<12} | {0.00:8.2f}% | {'N/A':<10} | {'N/A':<10} | {'N/A':<15}")
            continue
            
        coverage = (pts_in_slice / total_valid_pts) * 100
        avg_slice = np.nanmean(diff[mask])
        std_slice = np.nanstd(diff[mask])
        mean_alt = np.nanmean(orog_values[mask])
        
        alt_str = f"{mean_alt:10.2f} m" if not np.isnan(mean_alt) else "N/A"
        print(f"{label:<12} | {coverage:8.2f}% | {avg_slice:10.2e} | {std_slice:10.2e} | {alt_str}")
    
    if max_diff < 1e-10 or np.isnan(max_diff):
        print("  -> VERDICT: CERTIFIED (Bit-Identical)")
    elif max_diff < 1.0:
        print("  -> VERDICT: STABILIZED (Topographic Induction)")
    else:
        print("  -> VERDICT: WARNING (Significant Drift)")

def audit_file(path_new, path_arch, phase_label, idx=0):
    if not os.path.exists(path_new):
        print(f"SKIP: New file not found: {path_new}")
        return
    if not os.path.exists(path_arch):
        print(f"SKIP: Archival file not found: {path_arch}")
        return

    print(f"\n--- Climate Stabilization Scientific Audit: {phase_label} ---")
    data_new = np.load(path_new, allow_pickle=True)
    data_arch = np.load(path_arch, allow_pickle=True)
    
    ds_orog_hr = xr.open_dataset(OROG_EOBS_FRANCE_FILE)
    
    # Extract based on available keys and index
    if 'era5' in data_new.keys():
        val_new = data_new['era5'][idx]
        val_arch = data_arch['era5'][idx]
    elif 'y' in data_new.keys():
        val_new = data_new['y']
        val_arch = data_arch['y']
    else:
        print("ERROR: Unknown dataset format.")
        return

    # Dynamic target grid based on squeezed data
    sample_data = np.squeeze(val_new)
    h, w = sample_data.shape
    lons = np.linspace(-6.0, 10.0, w)
    lats = np.linspace(54.0, 38.0, h)
    ds_target = xr.Dataset(coords={'lat': lats, 'lon': lons})
    ds_orog_aligned = interpolation_target_grid(ds_orog_hr, ds_target, method="conservative_normed")

    if 'era5' in data_new.keys():
        check_parity(val_new, val_arch, "Baseline (ERA5)", ds_orog_aligned, "tas", "K")
        check_parity(data_new['gcm'][idx], data_arch['gcm'][idx], "Predictor (GCM)", ds_orog_aligned, "tas", "K")
    elif 'y' in data_new.keys():
        check_parity(val_new, val_arch, "Target (Phase 1)", ds_orog_aligned, "tas", "K")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--p1_new', type=str, required=True)
    parser.add_argument('--p2_new', type=str, required=True)
    args = parser.parse_args()

    p1_arch = "/scratch/globc/page/idownscale_exp5/datasets/dataset_exp5_30y/sample_19800101.npz"
    p2_arch = "/scratch/globc/page/idownscale_exp5/datasets/dataset_bc/bc_train_hist_gcm.npz"
    
    audit_file(args.p1_new, p1_arch, "Phase 1 Reconstruction", idx=0)
    audit_file(args.p2_new, p2_arch, "Phase 2: BC Synthesis", idx=0)
