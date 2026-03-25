"""
Core logic for COST Action ES1102 (VALUE) validation metrics.

This module provides standardized implementations for marginal, temporal, 
and spatial diagnostics used in downscaling validation.
"""

import numpy as np
from scipy.stats import pearsonr, wasserstein_distance

def get_marginal_metrics(obs, pred):
    """Compute marginal distributional metrics."""
    return {
        'bias': np.nanmean(pred) - np.nanmean(obs),
        'std_ratio': np.nanstd(pred) / np.nanstd(obs),
        'q5_bias': np.nanquantile(pred, 0.05) - np.nanquantile(obs, 0.05),
        'q50_bias': np.nanquantile(pred, 0.50) - np.nanquantile(obs, 0.50),
        'q95_bias': np.nanquantile(pred, 0.95) - np.nanquantile(obs, 0.95),
        'wasserstein': wasserstein_distance(obs.flatten(), pred.flatten())
    }

def get_temporal_metrics(obs, pred):
    """Compute temporal persistence metrics."""
    def lag1_autocorr(x):
        x = x.flatten()
        if len(x) < 2: return np.nan
        return pearsonr(x[:-1], x[1:])[0]
    
    return {
        'autocorr_obs': lag1_autocorr(obs),
        'autocorr_pred': lag1_autocorr(pred),
        'autocorr_error': lag1_autocorr(pred) - lag1_autocorr(obs)
    }

def get_spatial_metrics(obs_mean_map, pred_mean_map):
    """Compute spatial coherence metrics comparing time-mean maps."""
    obs_flat = obs_mean_map.flatten()
    pred_flat = pred_mean_map.flatten()
    
    # Mask NaNs for correlation
    mask = ~np.isnan(obs_flat) & ~np.isnan(pred_flat)
    if not np.any(mask):
        return {'spatial_corr': np.nan, 'spatial_rmse': np.nan}
    
    return {
        'spatial_corr': pearsonr(obs_flat[mask], pred_flat[mask])[0],
        'spatial_rmse': np.sqrt(np.nanmean((pred_mean_map - obs_mean_map)**2))
    }

def get_spell_length(data, threshold, operator='>'):
    """Calculate mean spell length above/below a threshold."""
    # This is a simplified 1D version for temporal points
    if operator == '>':
        active = (data > threshold).astype(int)
    else:
        active = (data < threshold).astype(int)
    
    # Identify changes
    diff = np.diff(active, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    durations = ends - starts
    return np.mean(durations) if len(durations) > 0 else 0.0
