import pytest

from iriscc.datautils import Data


def test_resolve_source_file_raises_when_non_rcm_match_is_ambiguous(monkeypatch):
    data = Data("france")
    monkeypatch.setattr(data, "_resolve_source_files", lambda source_name, var, date=None, ssp=None: ["a.nc", "b.nc"])
    monkeypatch.setattr(data, "get_source_spec", lambda source_name: {"geometry": "gcm"})

    with pytest.raises(FileExistsError, match="Expected exactly one source file"):
        data._resolve_source_file("gcm_cnrm_cm6_1", "tas")
