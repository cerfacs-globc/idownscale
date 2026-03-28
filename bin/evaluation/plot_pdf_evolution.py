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
from iriscc.settings import CONFIG, GRAPHS_DIR

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
    parser = argparse.ArgumentParser(description="Plot PDF evolution: Raw -> BC -> AI")
    parser.add_argument('--exp', type=str, required=True, help='Experiment name')
    parser.add_argument('--ssp', type=str, default='ssp585', help='Scenario')
    parser.add_argument('--raw', type=str, required=True, help='Path to raw GCM file')
    parser.add_argument('--bc', type=str, required=True, help='Path to BC file')
    parser.add_argument('--ai', type=str, required=True, help='Path to AI prediction file')
    args = parser.parse_args()

    print(f"Loading data for {args.exp}...")
    data_raw = load_data(args.raw, args.exp)
    data_bc = load_data(args.bc, args.exp)
    data_ai = load_data(args.ai, args.exp)

    # Filter constants/missing values if any (though iriscc handles some)
    raw_flat = data_raw.flatten()
    bc_flat = data_bc.flatten()
    ai_flat = data_ai.flatten()

    _, ax = plt.subplots(figsize=(10, 6))
    
    data_list = [raw_flat, bc_flat, ai_flat]
    labels = ['Raw GCM', 'Bias Corrected (Ibicus)', 'AI Downscaled (UNet)']
    colors = ['gray', 'blue', 'red']
    
    plot_histogram(data_list, ax, labels, colors, "Temperature [K]")
    
    ax.set_title(f"PDF Evolution: {args.exp} ({args.ssp})", fontsize=16, weight='bold')
    
    output_dir = GRAPHS_DIR / 'metrics' / args.exp
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{args.exp}_pdf_evolution_{args.ssp}.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PDF evolution plot saved to: {output_path}")

if __name__ == '__main__':
    main()
