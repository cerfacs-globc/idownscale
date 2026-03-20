import shutil
import os

src = '/archive2/globc/garcia/idownscale/rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc'
dst = '/scratch/globc/page/idownscale_active/rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc'

print(f"Copying {src} to {dst}...")
try:
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print("Success!")
    else:
        print(f"Source file {src} not found.")
except Exception as e:
    print(f"Copy failed: {e}")
