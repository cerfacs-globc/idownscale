import numpy as np
import os
from pathlib import Path
import xarray as xr
import pandas as pd

BASELINE_FILE = "data/audit/baseline_zoeleila_bc.npz"
baseline = np.load(BASELINE_FILE)
# Since the NPZ only has data, we rely on the documented 8x13 France grid
# Northern anchor: 52N, Southern anchor: 40N

print("--- METHODICAL COORDINATE AUDIT ---")
PROJECT_ROOT = Path(__file__).parents[3].resolve()
import sys
sys.path.append(str(PROJECT_ROOT))
from iriscc.datautils import Data, crop_domain_from_ds

FRANCE_DOMAIN = [-6., 12., 40., 52.]
get_data = Data(domain=FRANCE_DOMAIN)
ds_gcm = get_data.get_gcm_dataset('tas', pd.Timestamp('1979-01-01'))

# 1. Test Ascending (South-at-Row-0)
ds_asc = crop_domain_from_ds(ds_gcm.sortby('lat', ascending=True), FRANCE_DOMAIN)
print(f"\n[Ascending] Row 0 Lat: {ds_asc.lat.values[0]:.2f}, Row -1 Lat: {ds_asc.lat.values[-1]:.2f}")
print(f"[Ascending] Row 0 Mean: {ds_asc.tas.values[0].mean():.2f} K")

# 2. Test Descending (North-at-Row-0)
ds_desc = crop_domain_from_ds(ds_gcm.sortby('lat', ascending=False), FRANCE_DOMAIN)
print(f"\n[Descending] Row 0 Lat: {ds_desc.lat.values[0]:.2f}, Row -1 Lat: {ds_desc.lat.values[-1]:.2f}")
print(f"[Descending] Row 0 Mean: {ds_desc.tas.values[0].mean():.2f} K")

print("\n[Archival Anchor] Row 0 Mean (Warm North): 280.55 K")
