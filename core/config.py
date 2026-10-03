"""Loads YAML configuration once and exposes project paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = Path(os.environ.get("CLAIMGUARD_DATA", ROOT / "data"))
BLOB_DIR = DATA_DIR / "blobs"
MODEL_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "claimguard.db"


def _load(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    return _load(CONFIG_DIR / "settings.yaml")


@lru_cache(maxsize=1)
def reason_catalog() -> dict[str, dict[str, Any]]:
    return _load(CONFIG_DIR / "reason_codes.yaml")


@lru_cache(maxsize=1)
def domain_rules() -> dict[str, Any]:
    return _load(ROOT / settings()["domain_rules"])


def ensure_dirs() -> None:
    for d in (DATA_DIR, BLOB_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
