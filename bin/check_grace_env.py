import sys
import torch
import platform
import os

print(f"--- Environment Check ---")
print(f"Python version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print(f"Platform: {platform.platform()}")
print(f"Machine: {platform.machine()}")
print(f"Process ID: {os.getpid()}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device Count: {torch.cuda.device_count()}")
else:
    print("WARNING: CUDA is not available. Check if you are running on a GPU node and using --nv with Singularity.")
print(f"--- ---")
