import argparse
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Add current directory to path for iriscc imports
sys.path.append('.')

from iriscc.plotutils import plot_histogram
from iriscc.settings import CONFIG, GRAPHS_DIR, TARGET_EOBS_FRANCE_FILE

def load_data(path, exp):
    ds = xr.open_dataset(path)
    # Ensure standard longitude/domain for comparison
    if 'lon' in ds.coords and (ds.lon.max() > 180):
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    
    domain = CONFIG[exp]['domain']
    ds_cropped = ds.sel(lon=slice(domain[0], domain[1]), lat=slice(domain[2], domain[3]))
    data = ds_cropped.tas.values
    ds.close()
    return data

def main():
    parser = argparse.ArgumentParser(description="Plot PDF evolution: 5-Curve Master Validator")
    parser.add_argument('--exp', type=str, required=True, help='Experiment name')
    parser.add_argument('--ssp', type=str, default='ssp585', help='Scenario')
    parser.add_argument('--raw', type=str, required=True, help='Path to raw GCM file')
    parser.add_argument('--bc', type=str, required=True, help='Path to BC file')
    parser.add_argument('--ai', type=str, required=True, help='Path to AI prediction file')
    parser.add_argument('--era5', type=str, help='Path to ERA5 reformat file (Optional)')
    parser.add_argument('--eobs', type=str, default=str(TARGET_EOBS_FRANCE_FILE), help='Path to EOBS target file')
    args = parser.parse_args()

    print(f"Loading data for {args.exp} (5-Curve Mode)...")
    data_raw = load_data(args.raw, args.exp)
    data_bc = load_data(args.bc, args.exp)
    data_ai = load_data(args.ai, args.exp)
    data_era5 = load_data(args.era5, args.exp)
    data_eobs = load_data(args.eobs, args.exp)

    # Flatten and filter
    data_list = [
        data_raw.flatten(),
        data_bc.flatten(),
        data_ai.flatten(),
        data_era5.flatten(),
        data_eobs.flatten()
    ]
    
    labels = [
        'Raw GCM (Coarse)', 
        'Bias Corrected (GCM-BC)', 
        'UNet Downscaled',
        'ERA5 (Reanalysis)',
        'E-OBS (Observations)'
    ]
    
    # Matching EGU slide color palette
    colors = ['gray', 'blue', 'red', 'green', 'black']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_histogram(data_list, ax, labels, colors, "Temperature [K]")
    
    ax.set_title(f"Scientific Validation PDF: {args.exp} vs Observers", fontsize=16, weight='bold')
    
    output_dir = GRAPHS_DIR / 'metrics' / args.exp
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{args.exp}_master_5curve_validation_{args.ssp}.png'
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Master 5-curve plot saved to: {output_path}")

if __name__ == '__main__':
    main()
