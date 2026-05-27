#!/bin/bash
# Probe a specific torch version on Grace and report torch.version.cuda.

set -euo pipefail

TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
ENV_PATH="${ENV_PATH:-/scratch/globc/page/idownscale_rerun/scratch/grace_envs/torch_probe_${TORCH_VERSION//./_}}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=torch_probe_${TORCH_VERSION//./}
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

TORCH_VERSION="${TORCH_VERSION}"
ENV_PATH="${ENV_PATH}"

module load python/gloenv3.12_arm
module load nvidia/cuda/12.4
unset PYTHONHOME
export PYTHONNOUSERSITE=1

mkdir -p "\$(dirname "${ENV_PATH}")"
rm -rf "${ENV_PATH}"
python3 -m venv "${ENV_PATH}"
source "${ENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 "torch==${TORCH_VERSION}"

python - <<'PY'
import sys

print("python", sys.executable)
try:
    import torch

    print("torch", torch.__version__)
    print("torch.version.cuda", torch.version.cuda)
    print("torch.cuda.is_available", torch.cuda.is_available())
    print("torch.cuda.device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device_name", torch.cuda.get_device_name(0))
except Exception as exc:
    print("probe_ERR", repr(exc))
PY
EOF
