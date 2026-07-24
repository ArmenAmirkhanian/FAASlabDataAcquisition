"""
_dialog_utils.py — Shared helper so file-picker dialogs across the Processing
scripts (trim.py, view.py, section_merge.py, ...) remember the last folder
used, persisted across runs in a small JSON file next to the scripts.
"""

import os
import json

_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".last_dir.json")


def get_last_dir():
    """Folder to open file dialogs in: the last one used, if it still exists."""
    try:
        with open(_STATE_FILE, "r") as f:
            last_dir = json.load(f).get("last_dir", "")
        if last_dir and os.path.isdir(last_dir):
            return last_dir
    except (OSError, ValueError):
        pass
    return os.path.dirname(os.path.abspath(__file__))


def set_last_dir(path):
    """Remember the folder containing `path` (a file or directory) for next time."""
    try:
        folder = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
        with open(_STATE_FILE, "w") as f:
            json.dump({"last_dir": folder}, f)
    except OSError:
        pass
