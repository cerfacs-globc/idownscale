import numpy as np
import os

def diag_npz(file_new, file_base):
    data_new = np.load(file_new)
    data_base = np.load(file_base)
    
    for key in ['x', 'y']:
        print(f"\n--- Key: {key} ---")
        val_new = data_new[key]
        val_base = data_base[key]
        
        for i in range(val_new.shape[0]):
            v_n = val_new[i].astype(float)
            v_b = val_base[i].astype(float)
            diff = np.abs(v_n - v_b)
            print(f"  Layer {i}:")
            print(f"    New  - Mean: {np.nanmean(v_n):.4f}, Std: {np.nanstd(v_n):.4f}")
            print(f"    Base - Mean: {np.nanmean(v_b):.4f}, Std: {np.nanstd(v_b):.4f}")
            print(f"    MAX ABS DIFF: {np.nanmax(diff):.4f}")
            print(f"    MEAN ABS DIFF: {np.nanmean(diff):.4f}")

if __name__ == "__main__":
    new_f = "/gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y/sample_19800101.npz"
    base_f = "/scratch/globc/page/idownscale_exp5/datasets/dataset_exp5_30y/sample_19800101.npz"
    diag_npz(new_f, base_f)
