#!/bin/bash
# setup_venv.sh: Generic venv setup for EGU Short Course

ENV_DIR="venv_idownscale"

echo "=== Creating Virtual Environment: $ENV_DIR ==="
python3 -m venv $ENV_DIR

echo "=== Activating Environment ==="
source $ENV_DIR/bin/activate

echo "=== Upgrading Build Tools ==="
pip install --upgrade pip setuptools wheel

echo "=== Installing Dependencies from requirements.txt ==="
pip install -r requirements.txt

echo "=== Setup Complete ==="
echo "WARNING: Binary dependencies like ESMF and SBCK may require system-level installation."
echo "If you encounter errors with xesmf, please use the setup_conda.sh instead."
echo "To activate the environment, run: source $ENV_DIR/bin/activate"
