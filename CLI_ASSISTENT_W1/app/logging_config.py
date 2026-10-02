"""Logging setup.

Diagnostics go to a log file, never to the terminal, so they do not interleave
with streamed chat output. User-facing messages are the CLI's job.
"""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(level: str, log_file: Path) -> None:
    """Send all application logs at ``level`` or above to ``log_file``."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # The HTTP stack logs every request at INFO/DEBUG; keep it to warnings
    # unless we are explicitly debugging our own code.
    for noisy in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
