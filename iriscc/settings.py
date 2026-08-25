"""
Dynamic settings loader.

By default this module exposes the canonical settings from
``iriscc.settings_base``. When ``IDOWNSCALE_SETTINGS_MODULE`` is set, it loads
that module instead so the workflow can run against an explicit alternate
settings file without editing imports across the codebase.
"""

from __future__ import annotations

import importlib
import os


DEFAULT_SETTINGS_MODULE = "iriscc.settings_base"


def _resolve_requested_settings_module() -> str:
    requested = os.getenv("IDOWNSCALE_SETTINGS_MODULE", DEFAULT_SETTINGS_MODULE).strip()
    if not requested or requested in {__name__, "iriscc.settings"}:
        return DEFAULT_SETTINGS_MODULE
    return requested


_loaded_module = importlib.import_module(_resolve_requested_settings_module())

for _name, _value in vars(_loaded_module).items():
    if _name.startswith("__") and _name not in {"__all__", "__doc__"}:
        continue
    globals()[_name] = _value

ACTIVE_SETTINGS_MODULE = _loaded_module.__name__
