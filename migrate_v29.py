import os
from pathlib import Path
import shutil

src = '/archive2/globc/garcia/idownscale/rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc'
dst = '/scratch/globc/page/idownscale_active/rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc'

print(f"Copying {src} to {dst}...")
try:
    src_path = Path(src)
    dst_path = Path(dst)
    if src_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print("Success!")
    else:
        print(f"Source file {src} not found.")
except Exception as e: # noqa: BLE001
    print(f"Copy failed: {e}")
