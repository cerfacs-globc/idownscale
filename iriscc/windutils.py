from __future__ import annotations

import numpy as np


def speed_from_components(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(u) + np.square(v))


def direction_from_components(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological wind-from direction in degrees clockwise from north."""
    return np.mod(270.0 - np.degrees(np.arctan2(v, u)), 360.0)
