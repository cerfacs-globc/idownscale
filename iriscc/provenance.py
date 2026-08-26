"""
Lightweight W3C PROV-JSON helpers for workflow provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from shutil import which
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
ENV_EXACT_NAMES = {
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "PYTHONHOME",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
}
ENV_PREFIXES = ("IDOWNSCALE_", "SLURM_")
IMPORTANT_PACKAGES = (
    "sbck",
    "numpy",
    "scipy",
    "xarray",
    "xesmf",
    "ibicus",
    "torch",
    "pytorch-lightning",
    "netCDF4",
)
PROVENANCE_SCHEMA = "idownscale-provenance/1.0"


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_ready(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except TypeError:
            pass
    return str(value)


def _safe_check_output(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    timeout: float = 2.0,
) -> str | None:
    try:
        output = subprocess.check_output(  # noqa: S603
            command,
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    stripped = output.strip()
    return stripped or None


def _sha256_hexdigest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_sha256_max_bytes() -> int:
    raw_value = os.getenv("IDOWNSCALE_PROV_SHA256_MAX_BYTES", str(10 * 1024 * 1024))
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 10 * 1024 * 1024


def describe_path(path: str | Path) -> dict[str, JsonValue]:
    resolved = Path(path)
    info: dict[str, JsonValue] = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if resolved.exists():
        stat = resolved.stat()
        info["is_file"] = resolved.is_file()
        info["is_dir"] = resolved.is_dir()
        info["size_bytes"] = int(stat.st_size)
        info["mtime"] = datetime.fromtimestamp(stat.st_mtime, UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        if resolved.is_file():
            max_bytes = provenance_sha256_max_bytes()
            if stat.st_size <= max_bytes:
                info["sha256"] = _sha256_hexdigest(resolved)
            else:
                info["sha256_skipped"] = f"file larger than {max_bytes} bytes"
    return info


def inventory_paths(entries: dict[str, object]) -> dict[str, JsonValue]:
    inventory: dict[str, JsonValue] = {}
    for label, value in entries.items():
        if isinstance(value, (list, tuple, set)):
            inventory[label] = [describe_path(item) for item in value]
        else:
            inventory[label] = describe_path(value)
    return inventory


def git_commit(cwd: str | Path | None = None) -> str | None:
    git_bin = which("git")
    if git_bin is None:
        return None
    return _safe_check_output([git_bin, "rev-parse", "HEAD"], cwd=cwd)


def git_branch(cwd: str | Path | None = None) -> str | None:
    git_bin = which("git")
    if git_bin is None:
        return None
    return _safe_check_output([git_bin, "branch", "--show-current"], cwd=cwd)


def git_dirty(cwd: str | Path | None = None) -> bool | None:
    git_bin = which("git")
    if git_bin is None:
        return None
    status = _safe_check_output([git_bin, "status", "--short"], cwd=cwd)
    if status is None:
        return False
    return bool(status.strip())


def git_status_short(cwd: str | Path | None = None, max_lines: int = 50) -> list[str] | None:
    git_bin = which("git")
    if git_bin is None:
        return None
    status = _safe_check_output([git_bin, "status", "--short"], cwd=cwd)
    if status is None:
        return []
    return status.splitlines()[:max_lines]


def git_context(cwd: str | Path | None = None) -> dict[str, JsonValue]:
    return {
        "commit": git_commit(cwd),
        "branch": git_branch(cwd),
        "dirty": git_dirty(cwd),
        "status_short": git_status_short(cwd),
    }


def command_line() -> dict[str, JsonValue]:
    argv = list(sys.argv)
    return {
        "argv": argv,
        "shell": shlex.join(argv),
    }


def environment_snapshot() -> dict[str, JsonValue]:
    snapshot: dict[str, JsonValue] = {}
    for name in sorted(os.environ):
        if name in ENV_EXACT_NAMES or any(name.startswith(prefix) for prefix in ENV_PREFIXES):
            snapshot[name] = os.environ[name]
    return snapshot


def settings_identity() -> dict[str, JsonValue]:
    requested = os.getenv("IDOWNSCALE_SETTINGS_MODULE", "")
    active = None
    module_file = None
    settings_module = sys.modules.get("iriscc.settings")
    if settings_module is not None:
        active = getattr(settings_module, "ACTIVE_SETTINGS_MODULE", None)
        if active:
            loaded_module = sys.modules.get(active)
            module_file = getattr(loaded_module, "__file__", None) if loaded_module is not None else None
    return {
        "requested_module": requested,
        "active_module": active,
        "active_module_file": module_file,
    }


def runtime_agent() -> dict[str, JsonValue]:
    return {
        "user": os.getenv("USER", ""),
        "hostname": socket.gethostname(),
        "python_bin": os.getenv("PYTHON_BIN", ""),
        "runtime_root": os.getenv("IDOWNSCALE_RUNTIME_ROOT", ""),
        "raw_dir": os.getenv("IDOWNSCALE_RAW_DIR", ""),
        "output_dir": os.getenv("IDOWNSCALE_OUTPUT_DIR", ""),
        "graphs_dir": os.getenv("IDOWNSCALE_GRAPHS_DIR", ""),
        "runs_dir": os.getenv("IDOWNSCALE_RUNS_DIR", ""),
        "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
        "slurm_job_name": os.getenv("SLURM_JOB_NAME", ""),
    }


def package_versions() -> dict[str, JsonValue]:
    versions: dict[str, JsonValue] = {}
    try:
        distributions = importlib_metadata.distributions()
    except (AttributeError, OSError, TypeError, ValueError):
        return versions
    for distribution in distributions:
        name = distribution.metadata.get("Name")
        if not name:
            continue
        versions[name] = distribution.version
    return dict(sorted(versions.items(), key=lambda item: item[0].lower()))


def runtime_software() -> dict[str, JsonValue]:
    packages = package_versions()
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": sys.implementation.name,
        "packages": packages,
        "important_packages": {name: packages[name] for name in IMPORTANT_PACKAGES if name in packages},
    }


def _sysconf_bytes(page_key: str, size_key: str) -> int | None:
    try:
        pages = os.sysconf(page_key)
        size = os.sysconf(size_key)
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(pages, int) or not isinstance(size, int):
        return None
    return pages * size


def gpu_inventory() -> list[JsonValue]:
    nvidia_smi = which("nvidia-smi")
    if nvidia_smi is None:
        return []
    output = _safe_check_output(
        [
            nvidia_smi,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        timeout=2.0,
    )
    if output is None:
        return []
    rows: list[JsonValue] = []
    for line in output.splitlines():
        name, memory_total, driver_version = [item.strip() for item in line.split(",", maxsplit=2)]
        rows.append(
            {
                "name": name,
                "memory_total": memory_total,
                "driver_version": driver_version,
            }
        )
    return rows


def runtime_resources() -> dict[str, JsonValue]:
    return {
        "cpu_count": os.cpu_count(),
        "mem_total_bytes": _sysconf_bytes("SC_PHYS_PAGES", "SC_PAGE_SIZE"),
        "mem_available_bytes": _sysconf_bytes("SC_AVPHYS_PAGES", "SC_PAGE_SIZE"),
        "slurm_cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK", ""),
        "slurm_job_cpus_per_node": os.getenv("SLURM_JOB_CPUS_PER_NODE", ""),
        "slurm_mem_per_node": os.getenv("SLURM_MEM_PER_NODE", ""),
        "slurm_mem_per_cpu": os.getenv("SLURM_MEM_PER_CPU", ""),
        "slurm_gpus": os.getenv("SLURM_GPUS", ""),
        "slurm_gres": os.getenv("SLURM_JOB_GRES", ""),
        "gpus": gpu_inventory(),
    }


def chunk_metadata(parameters: dict[str, object]) -> dict[str, JsonValue]:
    keys = [
        "chunk_days",
        "phase1_chunk_days",
        "sbck_mbcn_max_fit_samples",
        "max_fit_samples",
        "cell_start",
        "cell_end",
        "cells_per_worker",
        "block_output_dir",
        "predict_block_size",
        "n_jobs",
        "future_start_date",
        "future_end_date",
        "phase1_start_date",
        "phase1_end_date",
        "sample_start_date",
        "sample_end_date",
        "predict_start_date",
        "predict_end_date",
        "metrics_start_date",
        "metrics_end_date",
        "value_start_date",
        "value_end_date",
    ]
    return {key: json_ready(parameters[key]) for key in keys if key in parameters and parameters[key] is not None}


def build_run_id(script_name: str, start_time: str, parameters: dict[str, object]) -> str:
    payload = {
        "script_name": script_name,
        "start_time": start_time,
        "parameters": json_ready(parameters),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"idownscale-run-{digest[:12]}"


def classify_artifact(label: str) -> str:
    normalized = label.lower()
    artifact_patterns = (
        ("checkpoint", "checkpoint"),
        ("statistics", "statistics"),
        ("sample", "sample_dataset"),
        ("topograph", "topography"),
        ("elevation", "topography"),
        ("mask", "mask"),
        ("grid", "grid"),
        ("prediction", "prediction"),
        ("metrics", "metrics"),
        ("graph", "diagnostic_graphs"),
        ("plot", "diagnostic_graphs"),
        ("run", "run_directory"),
        ("dataset", "dataset"),
    )
    for pattern, artifact_kind in artifact_patterns:
        if pattern in normalized:
            return artifact_kind
    return "artifact"


def default_parameter_sources(parameters: dict[str, object], settings: dict[str, object]) -> dict[str, JsonValue]:
    return {
        "parameters": dict.fromkeys(parameters, "cli"),
        "settings": dict.fromkeys(settings, "resolved"),
    }


def merged_parameter_sources(
    parameters: dict[str, object],
    settings: dict[str, object],
    parameter_sources: dict[str, object] | None = None,
) -> dict[str, JsonValue]:
    sources = default_parameter_sources(parameters, settings)
    if not parameter_sources:
        return sources
    for section in ("parameters", "settings"):
        raw_section = parameter_sources.get(section)
        if isinstance(raw_section, dict):
            for key, value in raw_section.items():
                sources[section][str(key)] = json_ready(value)
    return json_ready(sources)


def infer_model_metadata(parameters: dict[str, object], settings: dict[str, object]) -> dict[str, JsonValue]:
    metadata: dict[str, JsonValue] = {}
    for key in (
        "model",
        "loss",
        "learning_rate",
        "dropout",
        "seed",
        "max_epoch",
        "batch_size",
        "channels",
        "in_channels",
        "output_norm",
        "output_range",
        "diffusion_num_samples",
        "prediction_frequency",
        "training_frequency",
        "statistics_dir",
        "sample_dir",
    ):
        if key in settings and settings[key] is not None:
            metadata[key] = json_ready(settings[key])
        elif key in parameters and parameters[key] is not None:
            metadata[key] = json_ready(parameters[key])
    checkpoint_value = settings.get("checkpoint_dir", parameters.get("checkpoint_dir"))
    if checkpoint_value is not None:
        metadata["checkpoint"] = describe_path(checkpoint_value)
    return metadata


def derive_artifact_relations(
    script_name: str,
    input_entities: dict[str, str],
    output_entities: dict[str, str],
) -> dict[str, JsonValue]:
    relations: dict[str, JsonValue] = {}
    for output_index, (output_label, output_entity_id) in enumerate(output_entities.items()):
        for input_index, (input_label, input_entity_id) in enumerate(input_entities.items()):
            relation_id = f"idownscale:wdf:{script_name}:{output_index}:{input_index}"
            relations[relation_id] = {
                "prov:generatedEntity": output_entity_id,
                "prov:usedEntity": input_entity_id,
                "idownscale:generated_role": output_label,
                "idownscale:used_role": input_label,
            }
    return relations


def build_prov_bundle(
    *,
    script_name: str,
    activity_type: str,
    start_time: str,
    end_time: str,
    parameters: dict[str, object],
    settings: dict[str, object],
    inputs: dict[str, object],
    outputs: dict[str, object],
    cwd: str | Path | None = None,
    parameter_sources: dict[str, object] | None = None,
    model_metadata: dict[str, object] | None = None,
) -> dict[str, JsonValue]:
    activity_id = f"idownscale:{script_name}:{start_time}"
    agent_id = "idownscale:runtime-agent"
    run_id = build_run_id(script_name, start_time, parameters)
    resolved_model_metadata = infer_model_metadata(parameters, settings)
    if model_metadata:
        resolved_model_metadata.update({str(key): json_ready(value) for key, value in model_metadata.items()})
    bundle: dict[str, JsonValue] = {
        "prefix": {
            "prov": "http://www.w3.org/ns/prov#",
            "idownscale": "https://github.com/cerfacs-globc/idownscale#",
        },
        "idownscale:provenance_schema": PROVENANCE_SCHEMA,
        "idownscale:run_id": run_id,
        "entity": {},
        "activity": {
            activity_id: {
                "prov:type": f"idownscale:{activity_type}",
                "prov:label": script_name,
                "prov:startTime": start_time,
                "prov:endTime": end_time,
                "idownscale:run_id": run_id,
                "idownscale:command": json_ready(command_line()),
                "idownscale:parameters": json_ready(parameters),
                "idownscale:parameter_sources": merged_parameter_sources(parameters, settings, parameter_sources),
                "idownscale:settings": json_ready(settings),
                "idownscale:settings_identity": json_ready(settings_identity()),
                "idownscale:git": json_ready(git_context(cwd)),
                "idownscale:chunking": json_ready(chunk_metadata(parameters)),
                "idownscale:model_metadata": json_ready(resolved_model_metadata),
            }
        },
        "agent": {
            agent_id: {
                "prov:type": "prov:SoftwareAgent",
                "prov:label": "idownscale runtime",
                "idownscale:runtime": json_ready(runtime_agent()),
                "idownscale:environment": json_ready(environment_snapshot()),
                "idownscale:resources": json_ready(runtime_resources()),
                "idownscale:software": json_ready(runtime_software()),
            }
        },
        "used": {},
        "wasGeneratedBy": {},
        "wasDerivedFrom": {},
        "wasAssociatedWith": {
            f"idownscale:assoc:{script_name}:{start_time}": {
                "prov:activity": activity_id,
                "prov:agent": agent_id,
            }
        },
    }

    input_entities: dict[str, str] = {}
    for index, (label, path) in enumerate(inputs.items()):
        entity_id = f"idownscale:{script_name}:input:{index}"
        bundle["entity"][entity_id] = {
            "prov:type": "idownscale:input",
            "prov:label": label,
            "idownscale:path": json_ready(path),
            "idownscale:artifact_kind": classify_artifact(label),
            "idownscale:details": describe_path(path),
        }
        input_entities[label] = entity_id
        bundle["used"][f"idownscale:used:{script_name}:{index}"] = {
            "prov:activity": activity_id,
            "prov:entity": entity_id,
            "prov:role": label,
        }

    output_entities: dict[str, str] = {}
    for index, (label, path) in enumerate(outputs.items()):
        entity_id = f"idownscale:{script_name}:output:{index}"
        bundle["entity"][entity_id] = {
            "prov:type": "idownscale:output",
            "prov:label": label,
            "idownscale:path": json_ready(path),
            "idownscale:artifact_kind": classify_artifact(label),
            "idownscale:details": describe_path(path),
        }
        output_entities[label] = entity_id
        bundle["wasGeneratedBy"][f"idownscale:wgb:{script_name}:{index}"] = {
            "prov:entity": entity_id,
            "prov:activity": activity_id,
        }

    bundle["wasDerivedFrom"] = derive_artifact_relations(script_name, input_entities, output_entities)

    return bundle


def print_resolved_context(
    *,
    script_name: str,
    parameters: dict[str, object],
    settings: dict[str, object],
    inputs: dict[str, object],
    outputs: dict[str, object],
) -> None:
    payload = {
        "script": script_name,
        "parameters": json_ready(parameters),
        "settings": json_ready(settings),
        "inputs": json_ready(inputs),
        "outputs": json_ready(outputs),
        "runtime": json_ready(runtime_agent()),
    }
    print("=== IDOWNSCALE RESOLVED CONTEXT START ===", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("=== IDOWNSCALE RESOLVED CONTEXT END ===", flush=True)


def write_provjson(path: str | Path, bundle: dict[str, JsonValue]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(bundle), indent=2, sort_keys=True) + "\n")
    return path
