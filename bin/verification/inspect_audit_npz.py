import numpy as np
import os

p1_new = "/scratch/globc/page/idownscale_output/audit_month/p1/sample_19800101.npz"
if os.path.exists(p1_new):
    data = np.load(p1_new)
    print(f"Keys: {list(data.keys())}")
    for key in data.keys():
        arr = data[key]
        print(f"--- Key: {key} ---")
        print(f"Shape: {arr.shape}")
        print(f"Dtype: {arr.dtype}")
        print(f"Range: [{np.nanmin(arr)}, {np.nanmax(arr)}]")
        print(f"NaN Count: {np.isnan(arr).sum()}")
else:
    print(f"ERROR: File not found: {p1_new}")
