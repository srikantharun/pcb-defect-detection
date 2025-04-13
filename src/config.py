"""
Configuration parameters for PCB defect detection.
"""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
PROMPTS_DIR = os.path.join(DATA_DIR, "prompts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(os.path.join(IMAGES_DIR, "normal"), exist_ok=True)
os.makedirs(os.path.join(IMAGES_DIR, "defective"), exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Default files
DEFAULT_CATEGORIES_FILE = os.path.join(PROMPTS_DIR, "defect_categories.json")

# Model settings
DEFAULT_MODEL = "openai/clip-vit-base-patch32"
ALTERNATIVE_MODELS = [
    "openai/clip-vit-large-patch14",
    "facebook/flava-full",
    "google/siglip-base-patch16-224"
]

# Detection settings
DEFAULT_THRESHOLD = 0.2
DEFAULT_TOP_K = 3

# Visualization settings
VISUALIZATION_DPI = 300
DEFECT_COLORS = {
    "Normal": "green",
    "default": "red"
}

# Logging settings
LOGGING_LEVEL = "INFO"

# Batch processing settings
BATCH_SIZE = 16
