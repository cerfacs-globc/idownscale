"""
Audit Jan 31, 1980: compare new output vs archival anchor.
Reports avg, std, bias, maxdiff for all variables, all points.

Usage: ./bin/run_grace.sh bin/verification/audit_jan31.py
"""
import sys
sys.path.append('.')
import numpy as np

P1_NEW  = "/scratch/globc/page/idownscale_output/audit_month/p1/sample_19800131.npz"
P1_ARCH = "/scratch/globc/page/idownscale_exp5/datasets/dataset_exp5_30y/sample_19800131.npz"

def report(label, new_arr, arch_arr):
    mask = ~np.isnan(arch_arr)
    diff = new_arr[mask] - arch_arr[mask]
    print(f"  {label:<20} | avg={np.mean(new_arr[mask]):+.4e} | std={np.std(diff):.4e} | bias={np.mean(diff):+.4e} | maxdiff={np.max(np.abs(diff)):.4e}")

print("\n=== Audit: Jan 31, 1980 - Phase 1 ===")
try:
    new  = np.load(P1_NEW)
    arch = np.load(P1_ARCH)
    for key in arch.files:
        if key not in new:
            print(f"  {key}: MISSING in new file")
            continue
        a, b = arch[key].astype(np.float64), new[key].astype(np.float64)
        if a.ndim == 3:
            for c in range(a.shape[0]):
                report(f"{key}[ch{c}]", b[c], a[c])
        else:
            report(key, b, a)
except FileNotFoundError as e:
    print(f"  ERROR: {e}")

print("\n=== Audit: Phase 2 (bc_test_hist_gcm.npz) - Jan 31 entry ===")
P2_NEW  = "/scratch/globc/page/idownscale_output/datasets/dataset_bc/bc_test_hist_gcm.npz"
P2_ARCH = "/scratch/globc/page/idownscale_exp5/datasets/dataset_bc/bc_test_hist_gcm.npz"
AUDIT_DATE = "1980-01-31"
try:
    import pandas as pd
    new  = np.load(P2_NEW,  allow_pickle=True)
    arch = np.load(P2_ARCH, allow_pickle=True)
    dates_arch = pd.DatetimeIndex(arch['dates'])
    idx = np.where(dates_arch == pd.Timestamp(AUDIT_DATE))[0]
    if len(idx) == 0:
        print(f"  {AUDIT_DATE} not found in archival dates")
    else:
        i = idx[0]
        for key in ['era5', 'gcm']:
            if key in arch and key in new:
                report(key, new[key][i], arch[key][i])
except FileNotFoundError as e:
    print(f"  ERROR: {e}")
