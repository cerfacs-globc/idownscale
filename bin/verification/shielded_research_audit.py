import xarray as xr
import glob
import os

print("--- Forensic Phase 2 Metadata Audit ---")
files = glob.glob('rawdata/eobs/*.nc')
for f in files:
    try:
        ds = xr.open_dataset(f)
        print(f"FILE: {f}")
        print(f"  BOUNDS: LAT({ds.lat.min().values:.2f} to {ds.lat.max().values:.2f}), LON({ds.lon.min().values:.2f} to {ds.lon.max().values:.2f})")
        ds.close()
    except Exception as e:
        print(f"FILE: {f} -> ERROR: {e}")

# Searching for the missing protocol within the verification foundations
print("\n--- Protocol Logic Search ---")
os.system("grep -r 'OROG_EOBS_EUROPE_FILE' bin/verification/")
