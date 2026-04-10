# setup_conda.sh: Generic Conda environment setup for EGU Short Course

# Install in scratch space to avoid home quota issues
ENV_PATH="/scratch/globc/page/conda/envs/idownscale_egu"

echo "=== Creating Conda Environment in Scratch: $ENV_PATH ==="
conda create -y -p "$ENV_PATH" python=3.12 

echo "=== Activating Environment ==="
# Source conda.sh to ensure 'conda activate' works in the script
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

echo "=== Installing Binary Dependencies (ESMF/xesmf) via conda-forge ==="
conda install -y -c conda-forge esmf xesmf sbck cartopy pyproj

echo "=== Installing Python Dependencies via pip ==="
pip install -r requirements.txt

echo "=== Setup Complete ==="
echo "To activate the environment, run: conda activate $ENV_PATH"
