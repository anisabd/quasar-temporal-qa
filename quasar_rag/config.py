"""Project configuration: paths, model IDs, splits."""

from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
# Root data directory. Override with env var DATA_PATH if your data
# lives somewhere else (e.g. mounted Google Drive folder).
DATA_PATH: str = os.environ.get(
    "DATA_PATH",
    str(Path(__file__).resolve().parent.parent / "data"),
)

CACHE_DIR: str = os.path.join(DATA_PATH, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
LLM_MODEL_ID: str = "meta-llama/Llama-3.2-3B-Instruct"
EMBED_MODEL_NAME: str = "all-MiniLM-L6-v2"

# ----------------------------------------------------------------------
# Dataset splits
# ----------------------------------------------------------------------
SPLITS = ("train", "val", "test")
