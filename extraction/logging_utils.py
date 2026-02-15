"""Shared logging helpers for extraction scripts."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _normalize_level(level_name: Optional[str]) -> int:
    if not level_name:
        return logging.INFO
    return getattr(logging, level_name.strip().upper(), logging.INFO)


def setup_logging(log_file: Optional[Path] = None, level_name: Optional[str] = None) -> None:
    """Configure root logging once for both console and optional file output."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    level = _normalize_level(level_name or os.getenv("EXTRACTION_LOG_LEVEL"))
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=_LOG_FORMAT, handlers=handlers)

