"""
Lightweight W3C PROV-JSON helpers for workflow provenance.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return value


def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            or None
        )
    except Exception:
        return None


def runtime_agent() -> dict[str, Any]:
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


def build_prov_bundle(
    *,
    script_name: str,
    activity_type: str,
    start_time: str,
    end_time: str,
    parameters: dict[str, Any],
    settings: dict[str, Any],
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    activity_id = f"idownscale:{script_name}:{start_time}"
    agent_id = "idownscale:runtime-agent"
    bundle: dict[str, Any] = {
        "prefix": {
            "prov": "http://www.w3.org/ns/prov#",
            "idownscale": "https://github.com/cerfacs-globc/idownscale#",
        },
        "entity": {},
        "activity": {
            activity_id: {
                "prov:type": f"idownscale:{activity_type}",
                "prov:label": script_name,
                "prov:startTime": start_time,
                "prov:endTime": end_time,
                "idownscale:parameters": json_ready(parameters),
                "idownscale:settings": json_ready(settings),
                "idownscale:git_commit": git_commit(cwd),
            }
        },
        "agent": {
            agent_id: {
                "prov:type": "prov:SoftwareAgent",
                "prov:label": "idownscale runtime",
                "idownscale:runtime": json_ready(runtime_agent()),
            }
        },
        "used": {},
        "wasGeneratedBy": {},
        "wasAssociatedWith": {
            f"idownscale:assoc:{script_name}:{start_time}": {
                "prov:activity": activity_id,
                "prov:agent": agent_id,
            }
        },
    }

    for index, (label, path) in enumerate(inputs.items()):
        entity_id = f"idownscale:{script_name}:input:{index}"
        bundle["entity"][entity_id] = {
            "prov:type": "idownscale:input",
            "prov:label": label,
            "idownscale:path": json_ready(path),
        }
        bundle["used"][f"idownscale:used:{script_name}:{index}"] = {
            "prov:activity": activity_id,
            "prov:entity": entity_id,
            "prov:role": label,
        }

    for index, (label, path) in enumerate(outputs.items()):
        entity_id = f"idownscale:{script_name}:output:{index}"
        bundle["entity"][entity_id] = {
            "prov:type": "idownscale:output",
            "prov:label": label,
            "idownscale:path": json_ready(path),
        }
        bundle["wasGeneratedBy"][f"idownscale:wgb:{script_name}:{index}"] = {
            "prov:entity": entity_id,
            "prov:activity": activity_id,
        }

    return bundle


def print_resolved_context(
    *,
    script_name: str,
    parameters: dict[str, Any],
    settings: dict[str, Any],
    inputs: dict[str, Any],
    outputs: dict[str, Any],
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


def write_provjson(path: str | Path, bundle: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(bundle), indent=2, sort_keys=True) + "\n")
    return path
