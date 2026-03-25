"""
IRISCC package for climate downscaling.
"""

import pathlib

import torch

# Allow pathlib objects to be unpickled in modern PyTorch versions (2.4+)
# This resolves: _pickle.UnpicklingError: Unsupported global: pathlib.PosixPath
try:
    torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
except AttributeError:
    pass
