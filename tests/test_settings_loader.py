import importlib
import sys


def test_settings_loader_supports_env_override(monkeypatch, tmp_path):
    module_path = tmp_path / "custom_settings_test.py"
    module_path.write_text(
        "\n".join(
            [
                "CUSTOM_FLAG = 'ok'",
                "CONFIG = {'demo': {'phase1_chunk_days': 5}}",
                "def get_train_split_dates(exp):",
                "    return ['20000101', '20010101', '20020101']",
            ]
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("IDOWNSCALE_SETTINGS_MODULE", "custom_settings_test")

    sys.modules.pop("iriscc.settings", None)
    settings = importlib.import_module("iriscc.settings")

    assert settings.ACTIVE_SETTINGS_MODULE == "custom_settings_test"
    assert settings.CUSTOM_FLAG == "ok"
    assert settings.CONFIG["demo"]["phase1_chunk_days"] == 5

    sys.modules.pop("iriscc.settings", None)
