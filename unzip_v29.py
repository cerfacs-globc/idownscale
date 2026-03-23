import zipfile
import os
from pathlib import Path
import shutil

zip_path = '/scratch/globc/page/idownscale_active/6207c7b13d58afaec5f4a14b907de17e.zip'
extract_dir = '/scratch/globc/page/idownscale_active/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    files = zip_ref.namelist()
    print(f"Files in zip: {files}")
    zip_ref.extractall(extract_dir)

# Identify the elevation file (assuming it contains 'elev')
elev_files = [f for f in files if 'elev' in f.lower()]
if elev_files:
    src = Path(extract_dir) / elev_files[0]
    dst = '/scratch/globc/page/idownscale_active/rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc'
    print(f"Moving {src} to {dst}")
    shutil.move(src, dst)
else:
    print("No elevation file found in zip.")
