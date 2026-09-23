"""Project-wide paths and lightweight configuration.

Keep data and output paths in this module so that the rest of the project can
refer to them without duplicating filesystem assumptions.
"""

from pathlib import Path


# ``src/config.py`` -> project root is the parent of ``src``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Problem 1 source data discovered in the project workspace.
DATA_ROOT = PROJECT_ROOT / "data1"
LABEL_PATH = DATA_ROOT / "label-100.xlsx"

# All generated artifacts stay outside the raw-data directories.
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
METADATA_DIR = OUTPUT_ROOT / "metadata"
FIGURE_DIR = OUTPUT_ROOT / "figures"
LOG_DIR = OUTPUT_ROOT / "logs"


__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "LABEL_PATH",
    "OUTPUT_ROOT",
    "METADATA_DIR",
    "FIGURE_DIR",
    "LOG_DIR",
]
