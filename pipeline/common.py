"""Shared helpers for the pipeline: project paths and the settings from config.yaml.

Every pipeline step imports this, so paths and settings are defined in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


CONFIG = load_config()


def data_path(kind: str, filename: str = "") -> Path:
    """Return a path inside one of the data folders from config.yaml (input/interim/cache/output).

    The folder is created if it doesn't exist yet.
    """
    folder = ROOT / CONFIG["paths"][kind]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename if filename else folder
