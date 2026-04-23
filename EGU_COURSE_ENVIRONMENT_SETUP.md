# EGU 2026 Hands-on Workshop: Climate Downscaling Environment Setup

Welcome to the EGU presentation! This guide is designed for a **hands-on class**. Follow these steps to prepare your local machine (laptop).

---

## 🚀 Workshop Quick-Start: Laptop (macOS/WSL/Linux)
Most attendees will work on their personal machines. High-performance computing (HPC) is not required for the workshop tutorials.

1.  **Install Mamba**: [Miniforge](https://github.com/conda-forge/miniforge) is the recommended standard.
2.  **One-Command Environment Setup**:
    ```bash
    mamba create -n egu_workshop python=3.12 xarray xesmf cartopy pytorch-lightning -c conda-forge
    mamba activate egu_workshop
    pip install ibicus SBCK monai
    ```
3.  **The Workshop Orchestrator**:
    We have provided a machine-portable script that handles all data paths and execution.
    ```bash
    chmod +x bin/run_egu_hands_on_v86.sh
    ./bin/run_egu_hands_on_v86.sh
    ```

---

## 🛠️ Advanced Path: Supercomputer (Grace Hopper / ARM Native)
For professional production runs after the workshop:
1.  **Load the Production Stack**: `module load python/gloenv3.12_arm`
2.  **Enforce Isolation**: `export PYTHONNOUSERSITE=1`

---

## 🧪 The "Pre-flight" Health Check
Run this script to ensure your laptop's environment is healthy:

```python
import xarray as xr
import xesmf as xe
import torch

try:
    print(f"xarray version: {xr.__version__}")
    # Test Regridding engine
    ds_in = xr.Dataset({"lat": (["lat"], [30, 40]), "lon": (["lon"], [0, 10])})
    ds_out = xr.Dataset({"lat": (["lat"], [35]), "lon": (["lon"], [5])})
    regridder = xe.Regridder(ds_in, ds_out, method='bilinear')
    print("✅ Regridding engine: OK")
except Exception as e:
    print(f"❌ Setup Error: {e}")
```

---

## 💡 Troubleshooting & Performance
*   **The Noon/Midnight Trap**: Resolved in the master script via Integer Triple Matching.
*   **Windows (WSL)**: If you are on Windows, we strongly recommend using **WSL2** (Ubuntu) for the best compatibility with spatial libraries like `xesmf`.
*   **Coordinate Names**: ESMF expects `lat` and `lon`. The master script automatically audits these.
