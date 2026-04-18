#!/usr/bin/env python3
import os
import numpy as np
import argparse
from pathlib import Path

def validate(new_dir, ref_dir, num_samples=20):
    if not os.path.exists(new_dir):
        print(f"Error: New directory {new_dir} does not exist.")
        return
    if not os.path.exists(ref_dir):
        print(f"Error: Reference directory {ref_dir} does not exist.")
        return

    new_files = sorted([f for f in os.listdir(new_dir) if f.endswith('.npz')])
    ref_files = sorted([f for f in os.listdir(ref_dir) if f.endswith('.npz')])
    
    common = sorted(list(set(new_files).intersection(set(ref_files))))
    print(f"Found {len(new_files)} new files and {len(ref_files)} reference files.")
    print(f"Comparing {len(common)} common files...")
    
    if not common:
        print("Error: No common files found to compare!")
        return

    mismatches = 0
    checked = 0
    
    # Check a subset or all if small
    to_check = common if len(common) <= num_samples else common[:num_samples//2] + common[-num_samples//2:]
    
    for f in to_check:
        checked += 1
        new_path = os.path.join(new_dir, f)
        ref_path = os.path.join(ref_dir, f)
        
        try:
            new_data = np.load(new_path)
            ref_data = np.load(ref_path)
            
            file_ok = True
            for k in new_data.files:
                if k not in ref_data.files:
                    print(f"  [MISSING KEY] {f}: Key '{k}' not in reference.")
                    file_ok = False
                    continue
                
                new_val = new_data[k]
                ref_val = ref_data[k]
                
                diff = np.abs(new_val - ref_val).max()
                new_mean, ref_mean = np.mean(new_val), np.mean(ref_val)
                new_std, ref_std = np.std(new_val), np.std(ref_val)
                
                print(f"  [CHECK] {f} ({k}): MaxDiff={diff:.2e}, MeanDiff={abs(new_mean-ref_mean):.2e}, StdDiff={abs(new_std-ref_std):.2e}")
                
                if diff > 1e-7:
                    print(f"    -> [DISCREPANCY] {f}, Key: {k}, Max Diff: {diff:.2e}")
                    file_ok = False
                if abs(new_mean - ref_mean) > 1e-7:
                    print(f"    -> [MEAN DIFF] {f}, Key: {k}, Mean: {new_mean:.4f} vs {ref_mean:.4f}")
                    file_ok = False
            
            if not file_ok:
                mismatches += 1
        except Exception as e:
            print(f"  [ERROR] Failed to compare {f}: {e}")
            mismatches += 1

    print(f"\nSummary:")
    print(f"  Checked: {checked} files")
    print(f"  Mismatches: {mismatches}")
    
    if mismatches == 0:
        print("  SUCCESS: All checked samples match bit-perfectly (or within float tolerance).")
    else:
        print("  FAILURE: Found discrepancies in some samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset parity between two directories.")
    parser.add_argument("--new", type=str, required=True, help="Directory with newly generated samples")
    parser.add_argument("--ref", type=str, required=True, help="Directory with reference samples")
    parser.add_argument("--n", type=int, default=20, help="Number of samples to check")
    args = parser.parse_args()
    validate(args.new, args.ref, args.n)
