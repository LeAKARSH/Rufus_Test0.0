"""Logging setup for the scheduler/daemon and supporting modules."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rufus.config import Settings

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 5_000_000
_BACKUP_COUNT = 3


def setup_logging(settings: Settings | None = None, *, console: bool = True) -> None:
    """Configure root logging: a rotating file in ``log_dir`` plus (optionally)
    a console stream. The interactive shell calls with ``console=False`` so log
    lines never interleave with the REPL prompt."""
    settings = settings or Settings()
    root = logging.getLogger()
    if root.handlers:
        return

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
    root.setLevel(settings.log_level.upper())

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    log_dir: Path = settings.log_dir_path
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "rufus.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)