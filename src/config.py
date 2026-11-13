"""Central configuration for reproducible, memory-conscious model building."""
from dataclasses import dataclass
from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"


@dataclass(frozen=True)
class Settings:
    dataset_name: str = "CShorten/ML-ArXiv-Papers"
    split: str = "train"
    max_documents: int = int(os.getenv("RESEARCHRADAR_MAX_DOCUMENTS", "0"))
    lda_documents: int = int(os.getenv("RESEARCHRADAR_LDA_DOCUMENTS", "20000"))
    n_topics: int = int(os.getenv("RESEARCHRADAR_N_TOPICS", "10"))
    max_features: int = int(os.getenv("RESEARCHRADAR_MAX_FEATURES", "100000"))
    title_weight: float = float(os.getenv("RESEARCHRADAR_TITLE_WEIGHT", "2.0"))
    batch_size: int = 2048


SETTINGS = Settings()
for directory in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
