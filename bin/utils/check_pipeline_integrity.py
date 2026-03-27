import sys
import argparse
import numpy as np
import xarray as xr
from pathlib import Path
import json
import pandas as pd

def check_phase1_2(exp):
    print(f"--- Validating Phases 1-2 (Datasets) for {exp} ---")
    dataset_dir = Path(f'datasets/dataset_{exp}')
    if not dataset_dir.exists():
        dataset_dir = Path(f'datasets/dataset_{exp}_30y') # fallback
    
    samples = list(dataset_dir.glob('sample_*.npz'))
    if not samples:
        return False, "No samples found"
    
    # Check a random sample
    s = np.load(samples[0])
    x, y = s['x'], s['y']
    
    results = {
        'Samples Found': len(samples),
        'X Shape': str(x.shape),
        'Y Shape': str(y.shape),
        'X Range (K)': f"{np.min(x[1]):.1f} - {np.max(x[1]):.1f}", # Index 1 is tas
        'Y Range (K)': f"{np.min(y):.1f} - {np.max(y):.1f}",
        'Status': 'PASS' if 200 < np.mean(y) < 320 else 'FAIL (Unphysical)'
    }
    return True, results

def check_phase3(exp, simu):
    print(f"--- Validating Phase 3 (Bias Correction) for {exp} ---")
    bc_dir = Path(f'datasets/dataset_bc/dataset_{exp}_test_{simu}_bc')
    if not bc_dir.exists():
        return False, f"BC directory {bc_dir} not found"
    
    samples = list(bc_dir.glob('sample_*.npz'))
    if not samples:
        return False, "No BC samples found"
    
    s = np.load(samples[0])
    x = s['x'] # Elevation at 0, TAS at 1
    
    results = {
        'BC Samples': len(samples),
        'X Range (K)': f"{np.min(x[1]):.1f} - {np.max(x[1]):.1f}",
        'Status': 'PASS' if 200 < np.nanmean(x[1]) < 320 else 'FAIL'
    }
    return True, results

def check_phase4(exp):
    print(f"--- Validating Phase 4 (Training) for {exp} ---")
    log_dir = Path(f'runs/{exp}/unet_all/lightning_logs')
    if not log_dir.exists():
        return False, "Log directory not found"
    
    # Check latest version
    versions = sorted([v for v in log_dir.iterdir() if v.is_dir()], key=lambda x: int(x.name.split('_')[1]))
    if not versions:
        return False, "No versions found"
    
    latest = versions[-1]
    ckpt = list((latest / 'checkpoints').glob('*.ckpt'))
    
    results = {
        'Latest Version': latest.name,
        'Checkpoint Found': 'YES' if ckpt else 'NO',
        'Status': 'PASS' if ckpt else 'FAIL'
    }
    return True, results

def check_phase5(exp, test_name, simu_test):
    print(f"--- Validating Phase 5 (Inference) for {exp} ---")
    inf_dir = Path(f'graph/prediction/{exp}/{test_name}/{simu_test}')
    if not inf_dir.exists():
        return False, f"Inference directory {inf_dir} not found"
    
    files = list(inf_dir.glob('tas_*.nc'))
    if not files:
        return False, "No prediction files found"
    
    ds = xr.open_dataset(files[0])
    v = ds['tas'].values
    
    results = {
        'Files Generated': len(files),
        'TAS Range (K)': f"{np.nanmin(v):.1f} - {np.nanmax(v):.1f}",
        'Status': 'PASS' if 200 < np.nanmean(v) < 320 else 'FAIL'
    }
    return True, results

def check_phase6(exp):
    print(f"--- Validating Phase 6 (Evaluation) for {exp} ---")
    out_dir = Path(f'output/{exp}')
    if not out_dir.exists():
        return False, "Output directory not found"
    
    report = out_dir / f'evaluation_report_{exp}_unet_all_ssp585.pdf'
    csv = out_dir / f'value_metrics_{exp}_unet_all.csv'
    
    results = {
        'Report Found': 'YES' if report.exists() else 'NO',
        'CSV Found': 'YES' if csv.exists() else 'NO',
        'Status': 'PASS' if report.exists() and csv.exists() else 'FAIL'
    }
    if csv.exists():
        df = pd.read_csv(csv)
        results['Overall Bias'] = f"{df['bias'].iloc[0]:.3f} K"
    
    return True, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, required=True)
    parser.add_argument('--exp', default='exp5')
    parser.add_argument('--simu', default='gcm')
    parser.add_argument('--test-name', default='unet_all')
    parser.add_argument('--simu-test', default='gcm_bc')
    args = parser.parse_args()
    
    success = False
    data = {}
    
    if args.phase in [1, 2]:
        success, data = check_phase1_2(args.exp)
    elif args.phase == 3:
        success, data = check_phase3(args.exp, args.simu)
    elif args.phase == 4:
        success, data = check_phase4(args.exp)
    elif args.phase == 5:
        success, data = check_phase5(args.exp, args.test_name, args.simu_test)
    elif args.phase == 6:
        success, data = check_phase6(args.exp)
    
    if success:
        df = pd.DataFrame([data])
        print("\nMinimal Validation Table:")
        try:
            print(df.to_markdown(index=False))
        except (ImportError, Exception): # noqa: BLE001
            print(df.to_string(index=False))
        if 'FAIL' in str(data):
            sys.exit(1)
    else:
        print(f"Validation FAILED: {data}")
        sys.exit(1)

if __name__ == "__main__":
    main()
