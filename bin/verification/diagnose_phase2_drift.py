import numpy as np
import os

new_path = '/scratch/globc/page/idownscale_output/audit/audit_bc_19800120.npz'
arch_path = '/scratch/globc/page/idownscale_exp5/datasets/dataset_exp5_30y/sample_19800120.npz'

if not os.path.exists(new_path):
    print("New sample missing.")
    exit(1)

d_new = np.load(new_path)
d_arch = np.load(arch_path)

x_new = d_new['x']
x_arch = d_arch['x'][1] # Induction layer

print(f"--- Phase 2 Forensic Audit ---")
print(f"New  | Shape: {x_new.shape} | Min: {np.nanmin(x_new):.2f} | Max: {np.nanmax(x_new):.2f} | Mean: {np.nanmean(x_new):.2f}")
print(f"Arch | Shape: {x_arch.shape} | Min: {np.nanmin(x_arch):.2f} | Max: {np.nanmax(x_arch):.2f} | Mean: {np.nanmean(x_arch):.2f}")

zeros_new = np.sum(x_new == 0)
nans_new = np.sum(np.isnan(x_new))
print(f"New  | Zeros: {zeros_new} | NaNs: {nans_new}")

diff = np.abs(x_new - 0) # Checking if it's mostly zero
mean_val = np.nanmean(x_new)
if mean_val < 1.0:
    print("CRITICAL: New sample appears to be uninitialized or zeroed out.")
