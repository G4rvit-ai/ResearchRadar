"""Inspect and cache the Hugging Face dataset without fitting models."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SETTINGS, PROCESSED_DIR
from src.dataset_loader import load_huggingface_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-documents", type=int, default=1000,
                        help="Bounded sample size; use 0 only when intentionally loading the full split.")
    args = parser.parse_args()
    if args.max_documents < 0:
        parser.error("--max-documents must be 0 or a positive integer")
    papers = load_huggingface_dataset(SETTINGS.dataset_name, SETTINGS.split,
                                      max_documents=args.max_documents)
    print("Columns: paper_id, title, abstract")
    print(f"Rows after cleaning: {len(papers):,}")
    papers.head(3).to_csv(PROCESSED_DIR / "sample.csv", index=False)
