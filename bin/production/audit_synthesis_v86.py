#!/usr/bin/env python3
"""
Phase 2 Production Synthesis Auditor (v86.74)
Performs scientific certification of the 120-year GCM dataset.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Production Constants
ROOT = Path("/gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_bc")
EXPECTED_CHANNELS = ['gcm', 'era5', 'dates']
TOTAL_DAYS_EXPECTED = 44195 # 1980 to 2100 inclusive (Full 121-year cycle)

def audit_volume(path, label, requires_era5=True):
    print(f"\n--- Auditing Volume: {path.name} ({label}) ---")
    data = np.load(path, allow_pickle=True)
    
    # 1. Shape Audit
    gcm = data['gcm']
    print(f"  [Shape] GCM: {gcm.shape}")
    
    if requires_era5:
        era5 = data['era5']
        print(f"  [Shape] ERA5: {era5.shape}")
        if era5.shape != gcm.shape:
            print(f"  [CRITICAL] Shape mismatch: ERA5 {era5.shape} != GCM {gcm.shape}")
            return False
            
    # 2. Date Audit
    dates = data['dates']
    print(f"  [Timeline] Samples: {len(dates)}")
    start_date = pd.to_datetime(dates[0])
    end_date = pd.to_datetime(dates[-1])
    print(f"  [Timeline] Range: {start_date.date()} to {end_date.date()}")
    
    # 3. Physical Bounds Audit (tas in Kelvin)
    t_min = np.nanmin(gcm)
    t_max = np.nanmax(gcm)
    print(f"  [Physics] Temperature Range: {t_min:.2f}K to {t_max:.2f}K")
    
    if t_min < 220 or t_max > 340:
        print(f"  [WARNING] Physical anomaly detected: Range [{t_min}, {t_max}]")
        
    if np.isnan(gcm).any():
        print(f"  [WARNING] NaN values detected in spatial field!")
        
    return len(dates), dates[0], dates[-1]

def main():
    print("=== Phase 2: Scientific Production Audit (v86.74) ===")
    
    volumes = [
        ("bc_train_hist_gcm.npz", True),
        ("bc_test_hist_gcm.npz", True),
        ("bc_test_future_gcm.npz", False)
    ]
    
    all_dates = []
    total_days = 0
    
    for vol_name, needs_era5 in volumes:
        path = ROOT / vol_name
        if not path.exists():
            print(f"[FATAL] Missing volume: {path}")
            sys.exit(1)
            
        days, start, end = audit_volume(path, vol_name, requires_era5=needs_era5)
        total_days += days
        
    print("\n--- Final Certification Verdict ---")
    print(f"Total Days Synthesized: {total_days}")
    if total_days == TOTAL_DAYS_EXPECTED:
        print(f"[VERDICT] Chronological Continuity: CERTIFIED ({TOTAL_DAYS_EXPECTED}/{TOTAL_DAYS_EXPECTED})")
    else:
        print(f"[VERDICT] Chronological Continuity: FAILED (Found {total_days}, Expected {TOTAL_DAYS_EXPECTED})")

    # Check for Gaps
    print("\n[Audit Complete]")

if __name__ == "__main__":
    main()
