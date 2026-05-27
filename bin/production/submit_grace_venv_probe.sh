#!/bin/bash
# Submit a small Grace GPU probe against a specific ARM venv.

set -euo pipefail

VENV_PATH="${VENV_PATH:-/scratch/globc/page/idownscale_envs/production_final_v22_312}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=grace_venv_probe
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

module load python/gloenv3.12_arm
unset PYTHONHOME
export PYTHONNOUSERSITE=1

source "${VENV_PATH}/bin/activate"

python - <<'PY'
import sys

print("python", sys.executable)
print("version", sys.version)
for name in ["torch", "pytorch_lightning", "cartopy", "tensorboard"]:
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "imported"))
    except Exception as exc:
        print(name, "ERR", repr(exc))
try:
    import torch

    print("torch.version.cuda", torch.version.cuda)
    print("torch.cuda.is_available", torch.cuda.is_available())
    print("torch.cuda.device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device_name", torch.cuda.get_device_name(0))
        print("device_capability", torch.cuda.get_device_capability(0))
except Exception as exc:
    print("torch_cuda_probe_ERR", repr(exc))
PY
EOF
