#!/bin/bash
#SBATCH -p grace
#SBATCH --gres=gpu:0
#SBATCH --job-name=opt_bench
#SBATCH --output=/tmp/bench_opt1_%j.out
#SBATCH --error=/tmp/bench_opt1_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1

# Benchmark: OPT-1 (regridder cache) vs baseline serial throughput on 50 dates
# Runs inside the isolated idownscale_opt clone on an ARM grace node.

set -e
cd /scratch/globc/page/idownscale_opt || exit 1

PYTHON="/scratch/globc/page/idownscale_envs/env_idownscale_arm/bin/python"
unset PYTHONHOME; unset PYTHONPATH
export PYTHONNOUSERSITE=1
VENV_PATH=$(dirname "$(dirname "$PYTHON")")
export ESMFMKFILE="$VENV_PATH/lib/esmf.mk"
export PYTHONUNBUFFERED=1

echo "=== OPT-1 Benchmark ==="
echo "Start: $(date)"

# Phase 1: Run 50 dates with OPT-1 (regridder cache active)
$PYTHON - << 'PYEOF'
import sys, time, json, datetime
import pandas as pd

sys.path.insert(0, '/scratch/globc/page/idownscale_opt')

from iriscc.datautils import _REGRIDDER_CACHE
from bin.preprocessing.build_dataset import DatasetBuilder

DATES = pd.date_range('1990-01-01', periods=50, freq='D').tolist()
builder = DatasetBuilder('exp5')

print(f"[OPT-1] Running {len(DATES)} dates...", flush=True)
t0 = time.monotonic()
ok = 0
for date in DATES:
    try:
        builder.process_date(date, plot=False, baseline=False, force=False)
        ok += 1
    except Exception as e:
        print(f"  SKIP {date.date()}: {e}", flush=True)

elapsed = time.monotonic() - t0
rate_opt = ok / elapsed * 60
cache_entries = len(_REGRIDDER_CACHE)

print(f"[OPT-1] {ok}/50 in {elapsed:.1f}s = {rate_opt:.1f} dates/min, cache_entries={cache_entries}", flush=True)

# Phase 2: Run same 50 dates WITHOUT cache to simulate baseline
import importlib
for mod_name in list(sys.modules.keys()):
    if 'iriscc' in mod_name or 'bin.' in mod_name:
        del sys.modules[mod_name]

sys.path[0] = '/scratch/globc/page/idownscale_active'

from iriscc.datautils import interpolation_target_grid  # no cache version
from bin.preprocessing.build_dataset import DatasetBuilder as DatasetBuilderBaseline

builder2 = DatasetBuilderBaseline('exp5')
print(f"[BASELINE] Running {len(DATES)} dates...", flush=True)
t1 = time.monotonic()
ok2 = 0
for date in DATES:
    try:
        builder2.process_date(date, plot=False, baseline=False, force=False)
        ok2 += 1
    except Exception as e:
        print(f"  SKIP {date.date()}: {e}", flush=True)

elapsed2 = time.monotonic() - t1
rate_base = ok2 / elapsed2 * 60

print(f"[BASELINE] {ok2}/50 in {elapsed2:.1f}s = {rate_base:.1f} dates/min", flush=True)

speedup = rate_opt / rate_base if rate_base > 0 else 0
print(f"\n=== SPEEDUP OPT-1 vs BASELINE: {speedup:.2f}x ===", flush=True)

result = {
    'timestamp': datetime.datetime.utcnow().isoformat(),
    'opt1': {'ok': ok, 'elapsed_s': round(elapsed, 2), 'rate_dates_per_min': round(rate_opt, 1), 'cache_entries': cache_entries},
    'baseline': {'ok': ok2, 'elapsed_s': round(elapsed2, 2), 'rate_dates_per_min': round(rate_base, 1)},
    'speedup_x': round(speedup, 2),
}
with open('/tmp/bench_opt1_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print("Results written to /tmp/bench_opt1_result.json", flush=True)
PYEOF

echo "=== Benchmark DONE: $(date) ==="
