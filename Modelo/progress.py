"""Simple progress logging helpers for training scripts.

Provide small helpers to print timestamped stage/step messages and optionally
write them to a log file inside the `Modelo` folder. The API is intentionally
small so it can be imported and used in existing scripts without changing
behavior when not in verbose mode.

Functions:
- stage(name, msg=None): Print a high-level stage header.
- step(msg): Print a step-level message.
- get_logger(name): (future) placeholder for extension.

Usage:
    from Modelo.progress import stage, step
    stage("data_load", "Loading CSV files")
    step("Reading parameters.csv into DataFrame")

"""
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).parent
LOG_FILE = ROOT / "training_progress.log"

# small helpers

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_log(text: str, to_file: bool = True):
    if to_file:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except Exception:
            # if logging to file fails, ignore to avoid breaking training
            pass


def stage(name: str, msg: Optional[str] = None, to_file: bool = True):
    """Print a high-level stage marker with timestamp.

    Parameters
    - name: short stage name (e.g., 'data_load', 'training', 'evaluation')
    - msg: optional human-friendly message
    - to_file: whether to append to `Modelo/training_progress.log`
    """
    header = f"[{_timestamp()}] STAGE {name.upper()}: {msg or ''}"
    print(header, flush=True)
    _write_log(header, to_file=to_file)


def step(msg: str, to_file: bool = True):
    """Print a smaller step message under the current stage."""
    text = f"[{_timestamp()}]   - {msg}"
    print(text, flush=True)
    _write_log(text, to_file=to_file)


# convenience alias for interactive usage
log_stage = stage
log_step = step
