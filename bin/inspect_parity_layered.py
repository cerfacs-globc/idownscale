import numpy as np
import os

def compare_npz(file_new, file_base):
    data_new = np.load(file_new)
    data_base = np.load(file_base)
    
    for key in ['x', 'y']:
        print(f"\n--- Key: {key} ---")
        val_new = data_new[key]
        val_base = data_base[key]
        
        for i in range(val_new.shape[0]):
            v_n = val_new[i]
            v_b = val_base[i]
            diff = np.abs(v_n.astype(float) - v_b.astype(float))
            max_diff = np.nanmax(diff)
            print(f"  Layer {i}: Max Abs Diff = {max_diff:.2e}")
            if max_diff == 0:
                print(f"    BIT-IDENTICAL")

if __name__ == "__main__":
    new_f = "/gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y/sample_19800101.npz"
    base_f = "/scratch/globc/page/idownscale_exp5/datasets/dataset_exp5_30y/sample_19800101.npz"
    compare_npz(new_f, base_f)
