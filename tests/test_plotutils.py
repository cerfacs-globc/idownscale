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


def test_looks_like_geographic_extent_rejects_projected_domain():
    assert plotutils.looks_like_geographic_extent([60000.0, 1196000.0, 1617000.0, 2681000.0]) is False


def test_looks_like_geographic_extent_accepts_lon_lat_domain():
    assert plotutils.looks_like_geographic_extent([-5.0, 11.0, 41.0, 51.0]) is True
