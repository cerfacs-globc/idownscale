"""
Algorithm Registry for Bias Correction (Debiasers) and AI Models.

This module provides a unified interface to load different algorithms based on
configuration strings, enabling experiment modularity.
"""
import importlib
import logging
from typing import Any, Type

logger = logging.getLogger(__name__)

def get_debiaser(name: str) -> Type[Any]:
    """
    Dynamically loads a debiaser class from supported libraries.

    Checks in order:
    1. ibicus.debias (case-insensitive)
    2. SBCK (if installed)
    3. Custom registry mappings
    """
    # 1. Try Ibicus
    try:
        ibicus_debias = importlib.import_module('ibicus.debias')
        for attr in dir(ibicus_debias):
            if attr.lower() == name.lower():
                return getattr(ibicus_debias, attr)
    except ImportError:
        logger.debug("Ibicus not found.")

    # 2. Try SBCK
    try:
        # Note: SBCK often uses submodules like SBCK.METHD
        sbck = importlib.import_module('SBCK')
        # This is a placeholder for how SBCK might be structured
        if hasattr(sbck, name):
            return getattr(sbck, name)
    except ImportError:
        logger.debug("SBCK not found.")

    msg = f"Debiaser '{name}' could not be found in Ibicus or SBCK."
    raise ValueError(msg)

def get_model(name: str) -> Type:
    """
    Returns the model class from the internal iriscc.models package.
    """
    name = name.lower()
    
    # Mapping for common aliases
    mapping = {
        'unet': 'unet.UNet',
        'denoising_unet': 'denoising_unet.DenoisingUNet',
        'swin2sr': 'swin2sr.Swin2SR',
        'cddpm': 'cddpm.CDDPM',
        'miniunet': 'miniunet.MiniUNet',
        'miniswinunetr': 'miniswinunetr.MiniSwinUNETR'
    }
    
    if name not in mapping:
        msg = f"Model '{name}' is not registered in iriscc.models."
        raise ValueError(msg)
    
    module_path, class_name = mapping[name].split('.')
    module = importlib.import_module(f'iriscc.models.{module_path}')
    return getattr(module, class_name)
