import pandas as pd

from bin.preprocessing.build_dataset_pp import grouped_dates_for_source


class DummyData:
    def _resolve_source_files(self, source_name, var, date, ssp=None):
        year = pd.Timestamp(date).year
        return [f"/mock/{source_name}/{var}_{year}.nc"]


def test_grouped_dates_for_source_splits_year_boundaries():
    data = DummyData()
    dates = pd.date_range("1984-12-30", "1985-01-02", freq="D")

    groups = grouped_dates_for_source(data, "cerra", "tas", dates, "ssp585")

    assert len(groups) == 2
    assert list(groups[0][1]) == list(pd.date_range("1984-12-30", "1984-12-31", freq="D"))
    assert list(groups[1][1]) == list(pd.date_range("1985-01-01", "1985-01-02", freq="D"))
