"""
Centralized logging configuration for Resume Screener.
"""
import logging
import sys
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / 'logs'
_initialized = False

def setup_logger(name: str = __name__) -> logging.Logger:
    global _initialized
    logger = logging.getLogger(name)

    if _initialized:
        return logger

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _LOG_DIR / 'app.log'

    fmt = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    _initialized = True
    logger.info(f"Logging initialized → {log_file}")
    return logger
