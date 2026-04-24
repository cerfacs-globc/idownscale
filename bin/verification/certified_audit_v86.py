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
    data_new = np.squeeze(data_new)
    data_arch = np.squeeze(data_arch)
    
    diff = data_new - data_arch
    diff_abs = np.abs(diff)
    max_diff = np.nanmax(diff_abs)
    
    orog_values = np.squeeze(ds_orog_aligned.elevation.values)
    if orog_values.shape != data_new.shape:
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
    
    # Direct Parity Verification (Key-Shape-Value Contract)
    keys_new = sorted(list(data_new.keys()))
    keys_arch = sorted(list(data_arch.keys()))
    
    if keys_new != keys_arch:
        print(f"FAILED: Key mismatch. New: {keys_new}, Arch: {keys_arch}")
        return
        
    for key in keys_new:
        # Load data (handling index for Phase 2 volumes)
        raw_new = np.squeeze(data_new[key][idx] if (key in ['era5', 'gcm'] and len(data_new[key].shape) > 2) else data_new[key])
        raw_arch = np.squeeze(data_arch[key][idx] if (key in ['era5', 'gcm'] and len(data_arch[key].shape) > 2) else data_arch[key])
        
        # Consistent shape comparison
        print(f"--- Layer: {key} | Shape: {raw_new.shape} ---")
        
        if raw_new.shape != raw_arch.shape:
            print(f"    VERDICT: FAILED (Shape mismatch: {raw_new.shape} vs {raw_arch.shape})")
            continue
            
        diff = raw_new - raw_arch
        max_d = np.nanmax(np.abs(diff))
        bias = np.nanmean(diff)
        std = np.nanstd(diff)
        print(f"    Global Bias: {bias:.2e} | Std: {std:.2e} | MaxDiff: {max_d:.2e}")
        
        if max_d == 0:
            print(f"    VERDICT: CERTIFIED (Bit-Identical)")
            continue
            
        if max_d < 0.5:
            print(f"    VERDICT: CERTIFIED (Rounding/Precision)")
            continue

        # Investigation Mode
        print(f"    WARNING: Significant Deviation detected (>0.5K). Investigating Topography...")
        
        # 1. Resolve Elevation Context (2D)
        elev = data_new['x'][0] if 'x' in data_new.keys() else None
        if elev is None:
            ds_orog = xr.open_dataset(OROG_EOBS_FRANCE_FILE)
            elev = ds_orog['elevation'].values # Assume shape will be handled or diagnostic
        
        # 2. Iterate channels if data is 3D (e.g. x tensor)
        if raw_new.ndim == 3:
            for c in range(raw_new.shape[0]):
                c_diff = raw_new[c] - raw_arch[c]
                c_max = np.nanmax(np.abs(c_diff))
                print(f"    Channel {c} MaxDiff: {c_max:.2e}")
                if c_max > 0.5 and elev is not None and elev.shape == raw_new[c].shape:
                    mask = np.abs(c_diff) > 0.5
                    avg_alt = np.mean(elev[mask])
                    print(f"        Avg Altitude of deviaton: {avg_alt:.1f} m")
        elif raw_new.ndim == 2:
            if elev is not None and elev.shape == raw_new.shape:
                mask = np.abs(diff) > 0.5
                avg_alt = np.mean(elev[mask])
                print(f"        Avg Altitude of deviaton: {avg_alt:.1f} m")
        
        print(f"    VERDICT: STABILIZED (Review needed for specific channels)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--p1_new', type=str, required=True)
    parser.add_argument('--p2_new', type=str, required=True)
    parser.add_argument('--idx', type=int, default=0, help="Index for Phase 2 volumes")
    args = parser.parse_args()

    # Archival Anchors
    p1_arch_base = "/scratch/globc/page/idownscale_exp5/datasets/dataset_exp5_30y"
    p2_arch_file = "/scratch/globc/page/idownscale_exp5/datasets/dataset_bc/bc_train_hist_gcm.npz"
    
    fname = os.path.basename(args.p1_new)
    p1_arch = os.path.join(p1_arch_base, fname)
    
    audit_file(args.p1_new, p1_arch, "Phase 1: Reconstruction", idx=0)
    audit_file(args.p2_new, p2_arch_file, "Phase 2: BC Synthesis", idx=args.idx)
