import os
from pathlib import Path

IMAGE_SIZE = (128, 128)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)
DATA_DIR = Path(__file__).parent.parent / "Garbage classification"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
RANDOM_STATE = 42


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
