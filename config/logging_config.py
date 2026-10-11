"""Central logging setup. Call setup_logging() once, at the start of every
script, before anything else runs.

Writes to BOTH the console (colored by level, for readability while watching
a run) and a timestamped file under logs/ (plain text, for later review —
"better to visualize" a completed run without re-scrolling a terminal).
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOGS_DIR = _PROJECT_ROOT / "logs"


class _ColorFormatter(logging.Formatter):
    """ANSI color per level for console output. Most modern terminals
    (Windows Terminal, PowerShell 7+, any Linux/Mac shell) render this fine.
    If your terminal shows raw escape codes instead of colors, that's a
    terminal capability issue, not a bug — the log content is unaffected."""
    COLORS = {
        logging.DEBUG: "\x1b[36m",     # cyan
        logging.INFO: "\x1b[32m",      # green
        logging.WARNING: "\x1b[33m",   # yellow
        logging.ERROR: "\x1b[31m",     # red
        logging.CRITICAL: "\x1b[41m",  # red background
    }
    RESET = "\x1b[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{super().format(record)}{self.RESET}"


def setup_logging(level: str = "INFO", log_to_file: bool = True) -> Path | None:
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)
    root_logger.handlers.clear()  # avoid duplicate handlers on repeated setup_logging() calls

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_ColorFormatter(fmt, datefmt=datefmt))
    root_logger.addHandler(console_handler)

    log_file_path = None
    if log_to_file:
        _LOGS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = _LOGS_DIR / f"run_{timestamp}.log"

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        root_logger.addHandler(file_handler)

    if resolved_level > logging.DEBUG:
        for noisy_logger in ("httpx", "httpcore", "urllib3", "PIL", "filelock", "openai"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level={level.upper()}"
                + (f", writing to {log_file_path}" if log_file_path else ""))
    return log_file_path
