#!/usr/bin/env python3
"""
Generic script to crop a NetCDF file to a specific geographical domain.
Can use predefined domains from settings.py or custom user-defined domains.

Author: Antigravity AI
Date: 19/03/2026
"""

import sys
import argparse
from pathlib import Path
import xarray as xr
import datetime

# Add project root to sys.path
sys.path.append('.')

from iriscc.datautils import crop_domain_from_ds, standardize_longitudes, standardize_dims_and_coords
from iriscc.settings import CONFIG

def main():
    parser = argparse.ArgumentParser(description="Crop a NetCDF file to a specific domain.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input NetCDF file path.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output NetCDF file path.")
    parser.add_argument("--exp", "-e", type=str, help="Experiment name to fetch domain from settings.py (e.g., exp5).")
    parser.add_argument("--domain", "-d", type=float, nargs=4, help="Manual domain: min_lon max_lon min_lat max_lat.")
    parser.add_argument("--standardize", action="store_true", help="Standardize dimensions and longitudes before cropping.")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)

    # Determine domain
    domain = None
    if args.exp:
        if args.exp in CONFIG:
            domain = CONFIG[args.exp]['domain']
            print(f"Using domain from experiment '{args.exp}': {domain}")
        else:
            print(f"Error: Experiment '{args.exp}' not found in settings.py.")
            sys.exit(1)
    elif args.domain:
        domain = args.domain
        print(f"Using manual domain: {domain}")
    else:
        print("Warning: No domain specified. The entire file will be saved (possibly standardized).")

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Opening {input_path}...")
    ds = xr.open_dataset(input_path)

    if args.standardize:
        print("Standardizing dimensions and longitudes...")
        ds = standardize_dims_and_coords(ds)
        ds = standardize_longitudes(ds)

    if domain:
        print(f"Cropping to domain {domain}...")
        ds = crop_domain_from_ds(ds, domain)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_path}...")
    ds.to_netcdf(output_path)
    print("Done!")

if __name__ == "__main__":
    main()
