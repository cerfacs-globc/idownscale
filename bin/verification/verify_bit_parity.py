import numpy as np
import os
import sys
import argparse
import xarray as xr
import xesmf as xe
import pandas as pd

def verify_parity(new_path, archival_path, label=""):
    if not os.path.exists(new_path):
        print(f"ERROR: Generated file missing: {new_path}")
        return
    
    print(f"\n--- Bit-Parity Certification: {label} ---")
    d_new = np.load(new_path)
    d_arch = np.load(archival_path)
    
    # 1. Structural Analysis
    # Zoé Archival Structure: x=(2, 64, 64), y=(64, 64)
    # x[0]=Elevation, x[1]=Induction (TAS ERA5), y=Target (TAS EOBS)
    
    keys_new = list(d_new.keys())
    print(f"Sample Keys: {keys_new}")
    
    # Grid definitions for regridding (France Domain)
    domain = [-6.0, 10.0, 38.0, 54.0] 
    def make_grid(shape):
        # shape is (nlat, nlon) or (size,)
        if isinstance(shape, int):
            nlat = nlon = shape
        else:
            nlat, nlon = shape[-2], shape[-1]
            
        lon = np.linspace(domain[0], domain[1], nlon)
        lat = np.linspace(domain[2], domain[3], nlat)
        return xr.Dataset({'lat': (['lat'], lat), 'lon': (['lon'], lon)})

    def audit_var(name, val_new, val_arch, target_parity=0.0):
        print(f"--- Auditing {name} ---")
        print(f"Shapes: New={val_new.shape}, Arch={val_arch.shape} | Types: New={val_new.dtype}, Arch={val_arch.dtype}")
        
        # Enforce Common Reference Grid (64x64) for Resolution-Independent Audit
        if val_new.shape != (64, 64):
            print(f"Regridding New {val_new.shape} -> 64x64...")
            grid_in = make_grid(val_new.shape)
            grid_out = make_grid(64)
            regridder = xe.Regridder(grid_in, grid_out, method='bilinear', reuse_weights=False)
            val_new = regridder(val_new)
            
        if val_arch.shape != (64, 64):
            print(f"Regridding Arch {val_arch.shape} -> 64x64...")
            grid_in_arch = make_grid(val_arch.shape)
            grid_out = make_grid(64)
            regridder_arch = xe.Regridder(grid_in_arch, grid_out, method='bilinear', reuse_weights=False)
            val_arch = regridder_arch(val_arch).values if hasattr(regridder_arch(val_arch), 'values') else regridder_arch(val_arch)
        
        print("Calculating Drift on Common 64x64 Grid...")
        # Ensure numpy arrays for calculation
        if hasattr(val_new, 'values'): val_new = val_new.values
        if hasattr(val_arch, 'values'): val_arch = val_arch.values
        
        diff = val_new - val_arch
        abs_diff = np.abs(diff)
        max_ae = np.nanmax(abs_diff)
        bias = np.nanmean(diff)
        std = np.nanstd(diff)
        
        # Spatial Forensics: Identify where the MaxDiff occurred
        idx = np.unravel_index(np.nanargmax(abs_diff), abs_diff.shape)
        grid_64 = make_grid(64)
        max_lat = grid_64.lat.values[idx[0]]
        max_lon = grid_64.lon.values[idx[1]]
        
        status = "CERTIFIED" if max_ae <= target_parity else "DRIFT"
        if target_parity > 0 and max_ae <= target_parity: status = "CERTIFIED"
        
        print(f"{name:15} | MaxDiff: {max_ae:8.2e} K | Bias: {bias:8.2e} K | Std: {std:8.2e} | @({max_lat:5.2f}N, {max_lon:5.2f}E) | STATUS: {status}")

    # Variables for Audit
    if 'audit_bc' in new_path:
        x_new = d_new['x']
        y_new = d_new['y']
        
        # Archival Handling for Phase 2 (BC Training Archive)
        if 'dates' in d_arch:
            target_date_str = "1980-01-20"
            print(f"Searching for {target_date_str} in archive ({len(d_arch['dates'])} entries)...")
            
            # Robust String-based Search
            idx = -1
            arch_dates = d_arch['dates']
            for i, d in enumerate(arch_dates):
                # Handle potential numpy.datetime64 or string formats
                if pd.Timestamp(d).strftime('%Y-%m-%d') == target_date_str:
                    idx = i
                    break
            
            if idx == -1:
                print(f"ERROR: Date {target_date_str} not found in archival archive.")
                sys.exit(1)
            
            print(f"Index Found: {idx}")
            x_arch = d_arch['era5'][idx]
            y_arch = d_arch['gcm'][idx]
        else:
            # Fallback for single-sample anchors
            x_arch = d_arch['x'][1] if d_arch['x'].ndim == 3 else d_arch['x']
            y_arch = d_arch['y']
            
        audit_var('Induction (X)', x_new, x_arch, target_parity=0.5) # Certified Artifact 
        audit_var('Target (Y)', y_new, y_arch, target_parity=0.0)
    else:
        # Phase 1: High-Res Reconstruction Audit
        # Elevation (index 0), ERA5 (index 1)
        audit_var('Elevation (X0)', d_new['x'][0], d_arch['x'][0], target_parity=0.0)
        audit_var('Induction (X1)', d_new['x'][1], d_arch['x'][1], target_parity=0.0)
        audit_var('Target (Y)', d_new['y'], d_arch['y'], target_parity=0.0)

    print("-" * 100)
    print("--- Audit Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=str, required=True)
    parser.add_argument("--archival", type=str, required=True)
    parser.add_argument("--label", type=str, default="Audit")
    args = parser.parse_args()
    
    verify_parity(args.new, args.archival, args.label)
