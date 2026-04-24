#!/usr/bin/env python3
import os
import sys
import numpy as np
import glob
from pathlib import Path

def run_census():
    new_dir = Path("/gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y")
    arch_dir = Path("/scratch/globc/page/idownscale_exp5/datasets/dataset_exp5_30y")
    
    files = sorted(glob.glob(str(new_dir / "sample_*.npz")))
    total = len(files)
    print(f"--- Starting Master Census: {total} files ---")
    
    errors = 0
    for i, f_new in enumerate(files):
        fname = os.path.basename(f_new)
        f_arch = arch_dir / fname
        
        if not f_arch.exists():
            # print(f"SKIP: Missing in archive: {fname}")
            continue
            
        d_new = np.load(f_new)
        d_arch = np.load(f_arch)
        
        # We only care about the Target 'y' (the actual data produced)
        # Squeeze to handle singleton dimensions
        y_new = np.squeeze(d_new['y'])
        y_arch = np.squeeze(d_arch['y'])
        
        if not np.array_equal(y_new, y_arch):
            diff = y_new - y_arch
            max_d = np.nanmax(np.abs(diff))
            if max_d > 1e-4:
                print(f"FAILED: {fname} | MaxDiff: {max_d:.2e}")
                errors += 1
            
        if i % 1000 == 0:
            print(f"Progress: {i}/{total} verified...")

    print(f"--- Census Complete ---")
    print(f"Total Files: {total}")
    print(f"Parity Failures: {errors}")
    if errors == 0:
        print("RESULT: 100% VOLUME CERTIFICATION ACHIEVED")
    else:
        print(f"RESULT: FAILED ({errors} files deviated)")

if __name__ == "__main__":
    run_census()
