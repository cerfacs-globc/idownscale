import numpy as np
import os
from pathlib import Path
import xarray as xr
import pandas as pd

# Load Archive Baseline
BASELINE_FILE = "data/audit/baseline_zoeleila_bc.npz"
baseline = np.load(BASELINE_FILE)
print("--- ARCHIVAL DOMAIN AUDIT ---")
# If coordinates are not in the NPZ, we infer from previous forensic reads
# Archive Row 0 Mean: 280.55 K (South), Row -1 Mean: 275.85 K (North)
# This certifies ASCENDING orientation (South-at-Row-0).

# Load Current Foundation Grid (Standard GCM France for Experiment 5)
PROJECT_ROOT = Path(__file__).parents[3].resolve()
from bin.preprocessing.build_dataset import Data
FRANCE_DOMAIN = [-6., 12., 40., 52.]
get_data = Data(domain=FRANCE_DOMAIN)
ds_gcm = get_data.get_gcm_dataset('tas', pd.Timestamp('1979-01-01'))

print("\n--- FOUNDATION DOMAIN AUDIT ---")
print(f"GCM Native Lats (First 5): {ds_gcm.lat.values[:5]}")
print(f"GCM Native Lons (First 5): {ds_gcm.lon.values[:5]}")

# Scientific Protocol Verification: Universal Ascending
ds_ascending = ds_gcm.sortby('lat')
print(f"\n[Protocol] Universal Ascending Sorted Lats: {ds_ascending.lat.values[:5]}")

# Surgical Clipping: Match the (8, 13) shape
from iriscc.datautils import crop_domain_from_ds
ds_france = crop_domain_from_ds(ds_ascending, FRANCE_DOMAIN)
print(f"\n[Protocol] Clipped France Shape: {ds_france.tas.values[0].shape}")
print(f"Clipped Lats: {ds_france.lat.values}")
