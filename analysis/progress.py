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
import os
from typing import Optional

# Use simple os.path paths instead of pathlib for broader compatibility
ROOT = os.path.dirname(__file__)
LOG_FILE = os.path.join(ROOT, "training_progress.log")

# small helpers

# Global verbosity (0=minimal, 1=normal, 2=verbose, 3=debug)
# Consumers can set progress.VERBOSITY = 2 to increase detail.
VERBOSITY = 1

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


def stage(name: str, msg: Optional[str] = None, to_file: bool = True, verbosity: Optional[int] = None, details: Optional[dict] = None):
    """Print a high-level stage marker with timestamp.

    Parameters
    - name: short stage name (e.g., 'data_load', 'training', 'evaluation')
    - msg: optional human-friendly message
    - to_file: whether to append to `Modelo/training_progress.log`
    """
    v = VERBOSITY if verbosity is None else verbosity
    header = f"[{_timestamp()}] STAGE {name.upper()}: {msg or ''}"
    # Print header for normal+ verbosity; always log header when to_file True
    if v >= 1:
        print(header, flush=True)
    _write_log(header, to_file=to_file)

    # When verbosity is high, show structured details (if provided)
    if details and v >= 2:
        try:
            # pretty-print details in a compact way
            for k, val in (details.items() if isinstance(details, dict) else []):
                line = f"[{_timestamp()}]   * {k}: {val}"
                if v >= 2:
                    print(line, flush=True)
                _write_log(line, to_file=to_file)
        except Exception:
            # ignore problems in diagnostics
            pass


def step(msg: str, to_file: bool = True, verbosity: Optional[int] = None, details: Optional[dict] = None):
    """Print a smaller step message under the current stage.

    Args:
        msg: human readable message
        to_file: write to the progress log
        verbosity: override global verbosity for this message
        details: optional dict with extra info printed only at higher verbosity
    """
    v = VERBOSITY if verbosity is None else verbosity
    text = f"[{_timestamp()}]   - {msg}"
    if v >= 1:
        print(text, flush=True)
    _write_log(text, to_file=to_file)

    if details and v >= 2:
        try:
            for k, val in (details.items() if isinstance(details, dict) else []):
                line = f"[{_timestamp()}]       > {k}: {val}"
                if v >= 2:
                    print(line, flush=True)
                _write_log(line, to_file=to_file)
        except Exception:
            pass


# convenience alias for interactive usage
log_stage = stage
log_step = step
