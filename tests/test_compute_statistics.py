from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


def load_compute_statistics_module():
    module_path = Path(__file__).resolve().parents[1] / "bin" / "preprocessing" / "compute_statistics.py"
    spec = spec_from_file_location("compute_statistics", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_update_statistics_skips_all_nan_input():
    module = load_compute_statistics_module()

    result = module.update_statistics(1.0, 2.0, 3, 4.0, 5.0, np.array([np.nan, np.nan]))

    assert result == (1.0, 2.0, 3, 4.0, 5.0)


def test_resolve_split_index_raises_with_available_range(tmp_path):
    module = load_compute_statistics_module()
    dataset = np.array(
        [
            str((tmp_path / "sample_20200504.npz").resolve()),
            str((tmp_path / "sample_20241231.npz").resolve()),
        ]
    )

    with pytest.raises(ValueError, match=r"Available sample range is 20200504\.\.20241231"):
        module.resolve_split_index(dataset, tmp_path, "20030101", "train")


def test_validate_split_order_requires_strictly_increasing_indices():
    module = load_compute_statistics_module()

    with pytest.raises(ValueError, match="Expected train < val < test"):
        module.validate_split_order(10, 10, 20)
