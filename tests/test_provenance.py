from types import SimpleNamespace

from iriscc.provenance import (
    build_prov_bundle,
    chunk_metadata,
    command_line,
    describe_path,
    environment_snapshot,
    inventory_paths,
    package_versions,
    runtime_software,
    runtime_resources,
    settings_identity,
)


def test_describe_path_reports_existing_file_metadata(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("abc")
    info = describe_path(path)
    assert info["path"] == str(path)
    assert info["exists"] is True
    assert info["is_file"] is True
    assert info["size_bytes"] == 3
    assert "mtime" in info
    assert "sha256" in info


def test_describe_path_skips_sha256_for_large_files(monkeypatch, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("abc")
    monkeypatch.setenv("IDOWNSCALE_PROV_SHA256_MAX_BYTES", "2")
    info = describe_path(path)
    assert "sha256" not in info
    assert info["sha256_skipped"] == "file larger than 2 bytes"


def test_inventory_paths_supports_single_paths_and_lists(tmp_path):
    left = tmp_path / "left.txt"
    right = tmp_path / "right.txt"
    left.write_text("l")
    right.write_text("rr")
    inventory = inventory_paths({"single": left, "many": [left, right]})
    assert inventory["single"]["path"] == str(left)
    assert inventory["many"][0]["path"] == str(left)
    assert inventory["many"][1]["size_bytes"] == 2


def test_package_versions_collects_sorted_distribution_versions(monkeypatch):
    fake = [
        SimpleNamespace(metadata={"Name": "zeta"}, version="3.0"),
        SimpleNamespace(metadata={"Name": "Alpha"}, version="1.2"),
        SimpleNamespace(metadata={}, version="ignored"),
    ]
    monkeypatch.setattr("iriscc.provenance.importlib_metadata.distributions", lambda: fake)
    assert package_versions() == {"Alpha": "1.2", "zeta": "3.0"}


def test_runtime_software_reports_python_and_package_versions(monkeypatch):
    monkeypatch.setattr("iriscc.provenance.package_versions", lambda: {"sbck": "1.4.2"})
    software = runtime_software()
    assert software["python_version"]
    assert software["python_implementation"]
    assert software["packages"] == {"sbck": "1.4.2"}
    assert software["important_packages"] == {"sbck": "1.4.2"}


def test_command_line_reports_argv_and_shell(monkeypatch):
    monkeypatch.setattr("iriscc.provenance.sys.argv", ["python", "bin/run.py", "--exp", "exp_wind"])
    info = command_line()
    assert info["argv"] == ["python", "bin/run.py", "--exp", "exp_wind"]
    assert "--exp exp_wind" in info["shell"]


def test_environment_snapshot_filters_relevant_variables(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    conda_prefix = tmp_path / "conda"
    monkeypatch.setenv("IDOWNSCALE_RAW_DIR", str(raw_dir))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
    monkeypatch.setenv("UNRELATED_VAR", "skip")
    snapshot = environment_snapshot()
    assert snapshot["IDOWNSCALE_RAW_DIR"] == str(raw_dir)
    assert snapshot["SLURM_JOB_ID"] == "123"
    assert snapshot["CONDA_PREFIX"] == str(conda_prefix)
    assert "UNRELATED_VAR" not in snapshot


def test_settings_identity_reports_loaded_module(monkeypatch, tmp_path):
    settings_file = tmp_path / "settings_custom.py"
    fake_loaded = SimpleNamespace(__file__=str(settings_file))
    fake_settings = SimpleNamespace(ACTIVE_SETTINGS_MODULE="iriscc.settings_custom")
    monkeypatch.setenv("IDOWNSCALE_SETTINGS_MODULE", "iriscc.settings_custom")
    monkeypatch.setitem(__import__("sys").modules, "iriscc.settings", fake_settings)
    monkeypatch.setitem(__import__("sys").modules, "iriscc.settings_custom", fake_loaded)
    info = settings_identity()
    assert info["requested_module"] == "iriscc.settings_custom"
    assert info["active_module"] == "iriscc.settings_custom"
    assert info["active_module_file"] == str(settings_file)


def test_chunk_metadata_collects_known_keys():
    info = chunk_metadata(
        {
            "phase1_chunk_days": 30,
            "cell_start": 0,
            "cell_end": 300,
            "max_fit_samples": 1000,
            "unrelated": "skip",
        }
    )
    assert info == {"phase1_chunk_days": 30, "cell_start": 0, "cell_end": 300}


def test_runtime_resources_reports_stubbed_gpu_inventory(monkeypatch):
    monkeypatch.setattr("iriscc.provenance.os.cpu_count", lambda: 16)
    monkeypatch.setattr("iriscc.provenance._sysconf_bytes", lambda *args: 1024)
    monkeypatch.setattr("iriscc.provenance.gpu_inventory", lambda: [{"name": "GPU", "memory_total": "10 MiB", "driver_version": "1"}])
    info = runtime_resources()
    assert info["cpu_count"] == 16
    assert info["mem_total_bytes"] == 1024
    assert info["gpus"] == [{"name": "GPU", "memory_total": "10 MiB", "driver_version": "1"}]


def test_build_prov_bundle_embeds_runtime_software(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr("iriscc.provenance.runtime_software", lambda: {"python_version": "3.12.0", "packages": {"sbck": "1.4.2"}})
    monkeypatch.setattr("iriscc.provenance.command_line", lambda: {"argv": ["python", "run.py"], "shell": "python run.py"})
    monkeypatch.setattr("iriscc.provenance.settings_identity", lambda: {"requested_module": "iriscc.settings_custom"})
    monkeypatch.setattr("iriscc.provenance.git_context", lambda cwd: {"commit": "abc", "branch": "main", "dirty": False, "status_short": []})
    monkeypatch.setattr("iriscc.provenance.environment_snapshot", lambda: {"IDOWNSCALE_RAW_DIR": str(raw_dir)})
    monkeypatch.setattr("iriscc.provenance.runtime_resources", lambda: {"cpu_count": 8})
    bundle = build_prov_bundle(
        script_name="run_obs_workflow.py",
        activity_type="workflow",
        start_time="2026-08-26T00:00:00Z",
        end_time="2026-08-26T00:01:00Z",
        parameters={"exp": "exp_wind", "cell_start": 0, "cell_end": 300},
        settings={"training_frequency": "3h"},
        inputs={"bundle": tmp_path / "input.npz"},
        outputs={"result": tmp_path / "output.nc"},
    )
    agent = bundle["agent"]["idownscale:runtime-agent"]
    activity = bundle["activity"]["idownscale:run_obs_workflow.py:2026-08-26T00:00:00Z"]
    assert agent["idownscale:software"] == {"python_version": "3.12.0", "packages": {"sbck": "1.4.2"}}
    assert agent["idownscale:environment"] == {"IDOWNSCALE_RAW_DIR": str(raw_dir)}
    assert agent["idownscale:resources"] == {"cpu_count": 8}
    assert activity["idownscale:command"] == {"argv": ["python", "run.py"], "shell": "python run.py"}
    assert activity["idownscale:settings_identity"] == {"requested_module": "iriscc.settings_custom"}
    assert activity["idownscale:git"] == {"commit": "abc", "branch": "main", "dirty": False, "status_short": []}
    assert activity["idownscale:chunking"] == {"cell_start": 0, "cell_end": 300}
