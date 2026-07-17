import pytest
import numpy as np

from iriscc import plotutils


def test_resolve_plot_extent_prefers_explicit_extent():
    resolved = plotutils.resolve_plot_extent(
        domain=[0.0, 1000.0, 0.0, 1000.0],
        plot_extent=[-10.0, 10.0, 40.0, 55.0],
    )
    assert resolved == [-10.0, 10.0, 40.0, 55.0]


def test_resolve_plot_extent_uses_geographic_domain():
    resolved = plotutils.resolve_plot_extent(domain=[-8.0, 12.0, 41.0, 52.0])
    assert resolved == [-8.0, 12.0, 41.0, 52.0]


def test_resolve_plot_extent_falls_back_for_projected_domain():
    resolved = plotutils.resolve_plot_extent(domain=[60000.0, 1196000.0, 1617000.0, 2681000.0])
    assert resolved == plotutils.DEFAULT_FRANCE_EXTENT


def test_resolve_plot_extent_warns_on_projected_fallback():
    with pytest.warns(UserWarning, match="Falling back to the default France map extent"):
        resolved = plotutils.resolve_plot_extent(domain=[60000.0, 1196000.0, 1617000.0, 2681000.0])
    assert resolved == plotutils.DEFAULT_FRANCE_EXTENT


def test_looks_like_geographic_extent_rejects_projected_domain():
    assert plotutils.looks_like_geographic_extent([60000.0, 1196000.0, 1617000.0, 2681000.0]) is False


def test_looks_like_geographic_extent_accepts_lon_lat_domain():
    assert plotutils.looks_like_geographic_extent([-5.0, 11.0, 41.0, 51.0]) is True


@pytest.mark.parametrize(
    ("metric", "scale", "expected"),
    [
        ("rmse", "daily", np.arange(0.0, 7.0 + 0.5, 0.5)),
        ("rmse", "monthly", np.arange(0.0, 1.2 + 0.1, 0.1)),
        ("bias", "daily", np.arange(-4.0, 4.0 + 0.5, 0.5)),
        ("bias", "monthly", np.arange(-0.6, 0.6 + 0.1, 0.1)),
    ],
)
def test_get_shared_metric_levels_returns_expected_ranges(metric, scale, expected):
    levels = plotutils.get_shared_metric_levels(metric, scale)
    np.testing.assert_allclose(levels, expected)


def test_get_shared_metric_levels_returns_copy():
    levels = plotutils.get_shared_metric_levels("rmse", "daily")
    levels[0] = -999.0

    fresh_levels = plotutils.get_shared_metric_levels("rmse", "daily")
    assert fresh_levels[0] == 0.0


def test_get_shared_metric_levels_rejects_unknown_request():
    with pytest.raises(ValueError, match="Unsupported shared metric level request"):
        plotutils.get_shared_metric_levels("mae", "daily")
