# EGU 2026 Hands-on Workshop: Climate Downscaling Environment Setup

Welcome to the EGU presentation! This guide is designed for a **hands-on class**. Follow these steps to prepare your local machine (laptop) or prepare for deployment on your university's supercomputer.

---

## 🚀 Workshop Quick-Starts

### Path A: Laptop (Zero to Science in 5 mins)
Use this if you are working locally on Windows (WSL), macOS, or Linux.
1.  **Install Mamba**: [Miniforge](https://github.com/conda-forge/miniforge) is the standard.
2.  **One-Command Setup**:
    ```bash
    mamba create -n egu_workshop python=3.12 xarray xesmf cartopy pytorch-lightning -c conda-forge
    mamba activate egu_workshop
    pip install ibicus SBCK monai
    ```

### Path B: Supercomputer (High-Performance Path)
Use this if you have SSH access to a cluster (e.g., Slurm-based).
1.  **Pull the EGU Container**:
    ```bash
    # Standard PyTorch image with all CUDA drivers
    singularity pull docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
    ```
2.  **Verify GPU Linkage**:
    ```bash
    singularity run --nv pytorch_image.sif python -c "import torch; print(torch.cuda.is_available())"
    ```

---

## 🛠️ Essential "Big 3" Libraries
During the class, we will focus on these three pillars:
1.  **xarray**: The "Excel" of climate science. We use it to read `.nc` (NetCDF) files.
2.  **xESMF**: Handles spatial regridding. Crucial for moving from GCM resolution (100km) to local scales (8km).
3.  **SBCK / Ibicus**: Statistical engines that correct the "wet bias" or "cold bias" in raw climate models.

---

## 🧪 The "Pre-flight" Health Check
Before the class starts, run this script to ensure your environment is healthy. Copy-paste this into a file named `check_env.py`:

```python
import xarray as xr
import xesmf as xe
import torch

try:
    # 1. Test Data Loading
    print(f"xarray version: {xr.__version__}")
    
    # 2. Test Regridding (ESMF dependency)
    ds_in = xr.Dataset({"lat": (["lat"], [30, 40]), "lon": (["lon"], [0, 10])})
    ds_out = xr.Dataset({"lat": (["lat"], [35]), "lon": (["lon"], [5])})
    regridder = xe.Regridder(ds_in, ds_out, method='bilinear')
    print("✅ Regridding engine: OK")
    
    # 3. Test GPU
    if torch.cuda.is_available():
        print(f"✅ GPU Acceleration: OK ({torch.cuda.get_device_name(0)})")
    else:
        print("⚠️ GPU not found. Falling back to CPU mode.")
        
except Exception as e:
    print(f"❌ Setup Error: {e}")
```

---

## 💡 Troubleshooting Pittfalls
*   **Coordinate Names**: ESMF expects `lat` and `lon`. If your GCM data uses `latitude` or `nav_lat`, you **must** rename them:
    `ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})`
*   **Longitude Bounds**: Many models use 0-360 range. Our pipeline uses -180 to 180.
    `ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180`
*   **Memory Issues**: On laptops, use `ds.chunk()` to handle large datasets without crashing your RAM.
