"""Central configuration and project paths for WorkShield."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Data"
RAW_DATA_DIR = DATA_ROOT / "raw"
DATASET_DIR = DATA_ROOT / "rvf10k"
TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "Models"
RESULTS_DIR = PROJECT_ROOT / "Results"
MODEL_PATH = MODELS_DIR / "workshield_vgg16.keras"
METADATA_PATH = MODELS_DIR / "class_mapping.json"
HISTORY_PATH = RESULTS_DIR / "training_history.json"
TRAINING_PLOT_PATH = RESULTS_DIR / "training_history.png"
EVALUATION_JSON_PATH = RESULTS_DIR / "evaluation_results.json"
EVALUATION_REPORT_PATH = RESULTS_DIR / "evaluation_report.txt"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"

# Fixed class order: FAKE = 0, REAL = 1.
CLASS_NAMES = ("fake", "real")
CLASS_TO_INDEX = {"fake": 0, "real": 1}
INDEX_TO_CLASS = {0: "FAKE", 1: "REAL"}
IMAGE_SIZE = (224, 224)
# 16 is a memory-conscious starting point; set WORKSHIELD_BATCH_SIZE=32 if able.
BATCH_SIZE = int(os.environ.get("WORKSHIELD_BATCH_SIZE", "16"))
EPOCHS = int(os.environ.get("WORKSHIELD_EPOCHS", "20"))
RANDOM_SEED = 42
DEFAULT_THRESHOLD = 0.50
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def create_output_directories():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
