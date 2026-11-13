"""Inspect and cache the Hugging Face dataset without fitting models."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SETTINGS, PROCESSED_DIR
from src.dataset_loader import load_huggingface_dataset


if __name__ == "__main__":
    papers = load_huggingface_dataset(SETTINGS.dataset_name, SETTINGS.split)
    print("Columns: paper_id, title, abstract")
    print(f"Rows after cleaning: {len(papers):,}")
    papers.head(3).to_csv(PROCESSED_DIR / "sample.csv", index=False)
