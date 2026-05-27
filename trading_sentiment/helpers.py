"""
utils/helpers.py
~~~~~~~~~~~~~~~~
GoQuant — Fear & Greed Sentiment Engine

Shared utility functions used across the project:
environment variable loading, JSON I/O, CSV helpers, and logging setup.
"""

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with a consistent timestamp format.
    Call once at application startup.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Environment helpers ───────────────────────────────────────────────────────

def require_env(var: str) -> str:
    """
    Return the value of environment variable *var*.

    Raises:
        EnvironmentError: Variable is not set or is empty.
    """
    value = os.getenv(var, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{var}' is not set. "
            "Add it to your shell profile or a .env file."
        )
    return value


def load_dotenv(path: str | Path = ".env") -> None:
    """
    Minimal .env loader — sets KEY=VALUE pairs as environment variables.
    Does not override variables that are already set.

    Args:
        path: Path to the .env file (default: '.env' in current directory).
    """
    env_path = Path(path)
    if not env_path.is_file():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# ── JSON helpers ──────────────────────────────────────────────────────────────

def load_json(path: str | Path) -> Any:
    """
    Load and return parsed JSON from *path*.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object.

    Raises:
        FileNotFoundError: File does not exist.
        json.JSONDecodeError: File is not valid JSON.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: '{path}'")
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """
    Serialise *data* as JSON and write to *path* (creates parent dirs).

    Args:
        data:   Python object to serialise.
        path:   Destination file path.
        indent: JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def append_csv_row(path: str | Path, row: dict, fieldnames: list[str]) -> None:
    """
    Append *row* to a CSV file, writing a header on first creation.

    Args:
        path:       Destination CSV path (created if absent).
        row:        Dict mapping fieldname → value.
        fieldnames: Ordered list of column names.
    """
    path       = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.is_file()

    with open(path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def read_csv_as_dicts(path: str | Path) -> list[dict]:
    """
    Read an entire CSV file and return rows as a list of dicts.

    Args:
        path: Path to the CSV file.

    Returns:
        List of row dicts (empty list if file does not exist).
    """
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ── Timestamp helper ──────────────────────────────────────────────────────────

def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
