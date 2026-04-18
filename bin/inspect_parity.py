import numpy as np
import os
import sys

def compare_npz(file_new, file_base):
    print(f"Comparing:")
    print(f"  New:  {file_new}")
    print(f"  Base: {file_base}")
    
    if not os.path.exists(file_new):
        print(f"ERROR: New file not found.")
        return
    if not os.path.exists(file_base):
        print(f"ERROR: Baseline file not found.")
        return

    data_new = np.load(file_new)
    data_base = np.load(file_base)
    
    keys_new = sorted(data_new.files)
    keys_base = sorted(data_base.files)
    
    print(f"\nKeys in New:  {keys_new}")
    print(f"Keys in Base: {keys_base}")
    
    all_keys = set(keys_new) | set(keys_base)
    
    for key in sorted(all_keys):
        print(f"\n--- Key: {key} ---")
        if key not in data_new:
            print(f"  MISSING in New file.")
            continue
        if key not in data_base:
            print(f"  EXTRA in New file (not in base).")
            continue
            
        val_new = data_new[key]
        val_base = data_base[key]
        
        print(f"  Shape: New={val_new.shape}, Base={val_base.shape}")
        print(f"  Dtype: New={val_new.dtype}, Base={val_base.dtype}")
        
        if val_new.shape != val_base.shape:
            print(f"  WARNING: Shape mismatch!")
            continue
            
        try:
            diff = np.abs(val_new.astype(float) - val_base.astype(float))
            max_diff = np.nanmax(diff)
            mean_diff = np.nanmean(diff)
            print(f"  Max Abs Diff:  {max_diff:.2e}")
            print(f"  Mean Abs Diff: {mean_diff:.2e}")
            if max_diff == 0:
                print(f"  BIT-IDENTICAL for this key.")
        except Exception as e:
            print(f"  Could not compute diff: {e}")

if __name__ == "__main__":
    new_f = "/gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y/sample_19800101.npz"
    base_f = "/scratch/globc/page/idownscale_exp5/datasets/dataset_exp5_30y/sample_19800101.npz"
    compare_npz(new_f, base_f)
