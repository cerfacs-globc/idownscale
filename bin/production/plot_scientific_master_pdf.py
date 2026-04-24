#!/usr/bin/env python3
"""
EGU26 Signature Plot: 5-Curve PDF Scientific Validator
Stacks Raw GCM, BC-GCM, UNet, ERA5, and EOBS for absolute validation.
"""
import argparse
import sys
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path

# Add project root to path
sys.path.append('.')

from iriscc.settings import CONFIG, GRAPHS_DIR, COLORS, TARGET_EOBS_FRANCE_FILE, ERA5_REFORMAT_FILE
from iriscc.plotutils import plot_histogram
from iriscc.datautils import standardize_longitudes, crop_domain_from_ds

def load_and_standardize(path, exp):
    ds = xr.open_dataset(path)
    # 1. Standardize longitudes
    if 'lon' in ds.coords and (ds.lon.max() > 180):
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    
    # 2. Crop to domain
    domain = CONFIG[exp]['domain']
    ds = ds.sel(lon=slice(domain[0], domain[1]), lat=slice(domain[2], domain[3]))
    
    # 3. Handle Temp Conversion (Celsius to Kelvin if needed)
    vals = ds.tas.values
    if np.nanmean(vals) < 100: # Clearly Celsius
        vals += 273.15
        
    ds.close()
    return vals.flatten()

def main():
    parser = argparse.ArgumentParser(description="EGU26 5-Curve PDF Master")
    parser.add_argument('--exp', type=str, default='exp5', help='Experiment')
    parser.add_argument('--ssp', type=str, default='ssp585', help='Scenario')
    parser.add_argument('--raw', type=str, required=True, help='Path to raw GCM')
    parser.add_argument('--bc', type=str, required=True, help='Path to BC GCM')
    parser.add_argument('--ai', type=str, required=True, help='Path to AI prediction')
    args = parser.parse_args()

    print(f"--- Generating 5-Curve PDF Master for {args.exp} ---")
    
    # 1. Load the 5 Curves
    curves = {
        'E-OBS (Obs)': load_and_standardize(TARGET_EOBS_FRANCE_FILE, args.exp),
        'ERA5 (Rean)': load_and_standardize(ERA5_REFORMAT_FILE, args.exp),
        'Raw GCM': load_and_standardize(args.raw, args.exp),
        'GCM-BC': load_and_standardize(args.bc, args.exp),
        'UNet (SOTA)': load_and_standardize(args.ai, args.exp)
    }

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_list = [curves[k] for k in curves.keys()]
    labels = list(curves.keys())
    # Precise EGU Palette
    colors = ['black', 'green', 'gray', 'blue', 'red']
    
    plot_histogram(data_list, ax, labels, colors, "Temperature [K]")
    
    plt.title(f"EGU26 Validation: Distributional Parity ({args.exp})", fontsize=14, weight='bold')
    plt.grid(alpha=0.3)
    
    output_path = GRAPHS_DIR / 'metrics' / args.exp / f'{args.exp}_egu26_signature_pdf.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Signature Plot Saved: {output_path}")

if __name__ == "__main__":
    main()
