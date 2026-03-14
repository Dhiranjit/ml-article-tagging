from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data/raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data/processed"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
CONFIGS_DIR = PROJECT_ROOT / "configs"