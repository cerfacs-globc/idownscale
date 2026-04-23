import numpy as np
import os
import sys
import xarray as xr
import pandas as pd

# Internal logic alignment
sys.path.append(os.getcwd())
from iriscc.datautils import interpolation_target_grid
from iriscc.settings import OROG_EOBS_FRANCE_FILE

def check_parity(data_new, data_arch, layer_label, ds_orog_aligned, var_name="tas", units="K"):
    """Perform statistical parity audit with nan-aware topographic correlation."""
    diff = data_new - data_arch
    diff_abs = np.abs(diff)
    max_diff = np.max(diff_abs)
    total_pts = diff_abs.size
    
    # ExtractAligned Orography Values
    orog_values = ds_orog_aligned.elevation.values
    if orog_values.shape != data_new.shape:
        orog_values = np.zeros_like(data_new)

    print(f"\nAUDIT LAYER: {layer_label}")
    print(f"Overall Metrics: Variable: {var_name:8} | Units: {units:3} | Bias: {np.mean(diff):10.2e} | Std: {np.std(diff):10.2e} | MaxDiff: {max_diff:10.2e}")
    
    # Enhanced ASCII Evidence Table
    print(f"{'Threshold':<12} | {'Coverage':<10} | {'Avg Diff':<10} | {'Std Diff':<10} | {'Mean Altitude':<15}")
    print("-" * 75)
    
    # Thresholds: -1.0 means "Overall" (everything)
    for thresh in [-1.0, 0.1, 1.0, 5.0]:
        if thresh < 0:
            mask = np.ones_like(diff_abs, dtype=bool)
            label = "Overall"
        else:
            mask = diff_abs > thresh
            label = f">{thresh:<6}"
            
        pts_in_slice = np.sum(mask)
        if pts_in_slice == 0:
            print(f"{label:<12} | {0.00:8.2f}% | {'N/A':<10} | {'N/A':<10} | {'N/A':<15}")
            continue
            
        coverage = (pts_in_slice / total_pts) * 100
        avg_slice = np.mean(diff[mask])
        std_slice = np.std(diff[mask])
        
        # Use nanmean to avoid sea-pixels poisoning the altitude mean
        mean_alt = np.nanmean(orog_values[mask])
        alt_str = f"{mean_alt:10.2f} m" if not np.isnan(mean_alt) else "N/A (Sea)"
        
        print(f"{label:<12} | {coverage:8.2f}% | {avg_slice:10.2e} | {std_slice:10.2e} | {alt_str}")
    
    if max_diff < 1e-10:
        print("  -> VERDICT: CERTIFIED (Bit-Identical)")
    elif max_diff < 1.0:
        print("  -> VERDICT: STABILIZED (Topographic Induction)")
    else:
        print("  -> VERDICT: WARNING (Significant Drift)")

def audit_file(path_new, path_arch, phase_label):
    """Load and audit individual files with rigorously aggregated orography."""
    if not os.path.exists(path_new):
        print(f"SKIP: New file not found: {path_new}")
        return
    if not os.path.exists(path_arch):
        print(f"SKIP: Archival file not found: {path_arch}")
        return

    print(f"\n--- Climate Stabilization Scientific Audit: {phase_label} ---")
    data_new = np.load(path_new, allow_pickle=True)
    data_arch = np.load(path_arch, allow_pickle=True)
    
    # Load High-Res Reference Orography (E-OBS)
    ds_orog_hr = xr.open_dataset(OROG_EOBS_FRANCE_FILE)
    
    # 1. Determine Grid from Data
    if 'era5' in data_new.keys():
        sample_data = data_new['era5'][0]
    elif 'y' in data_new.keys():
        sample_data = data_new['y']
    else:
        print("ERROR: Unknown dataset format.")
        return

    # 2. Reconstruct Target Grid for xESMF (Domain exp5)
    h, w = sample_data.shape
    lons = np.linspace(-6.0, 10.0, w)
    lats = np.linspace(54.0, 38.0, h)
    ds_target = xr.Dataset(coords={'lat': lats, 'lon': lons})
    
    # 3. Perform Conservative Interpolation of Orography
    ds_orog_aligned = interpolation_target_grid(ds_orog_hr, ds_target, method="conservative_normed")

    # 4. Run Parity Audit
    if 'era5' in data_new.keys():
        # Target Index 30 (January 31, 1980) for Monthly Benchmark
        idx = 30 
        check_parity(data_new['era5'][idx], data_arch['era5'][idx], "Baseline (ERA5)", ds_orog_aligned, "tas", "K")
        check_parity(data_new['gcm'][idx], data_arch['gcm'][idx], "Predictor (GCM)", ds_orog_aligned, "tas", "K")
    elif 'x' in data_new.keys():
        check_parity(data_new['y'], data_arch['y'], "Target (Phase 1)", ds_orog_aligned, "tas", "K")

if __name__ == "__main__":
    p1_new = "/scratch/globc/page/idownscale_output/audit_month/p1/sample_19800131.npz"
    p1_arch = "/scratch/globc/page/idownscale_exp5/datasets/dataset_exp5_30y/sample_19800131.npz"
    
    p2_new = "/scratch/globc/page/idownscale_output/audit_month/p2/bc_train_hist_gcm.npz"
    p2_arch = "/scratch/globc/page/idownscale_exp5/datasets/dataset_bc/bc_train_hist_gcm.npz"
    
    # Audit logic handles index 30 for Jan 31 within multi-day files
    audit_file(p1_new, p1_arch, "Phase 1 Reconstruction")
    audit_file(p2_new, p2_arch, "Phase 2: BC Synthesis")
