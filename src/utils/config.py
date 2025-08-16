# notebooks/config.py
# This file contains common configuration variables for all notebooks.
# By defining them here, you ensure consistency and make updates easier.

# --- Path Definitions ---
# These paths are relative to the location of the notebooks.
# '../' means to go up one directory level from 'notebooks/' to the project root.

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR /  "models"
