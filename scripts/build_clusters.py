"""Build the optional MiniBatch K-Means experiment from the saved TF-IDF index."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clustering import fit_clusters, save_clusters
from src.config import MODELS_DIR
from src.tfidf_engine import TfidfIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=10)
    args = parser.parse_args()
    if args.clusters < 2:
        raise SystemExit("--clusters must be at least 2.")
    index = TfidfIndex.load(MODELS_DIR / "tfidf")
    model, labels = fit_clusters(index.matrix, args.clusters)
    output = MODELS_DIR / "clustering.joblib"
    save_clusters(model, labels, output)
    print(f"Built {args.clusters} clusters for {len(labels):,} papers: {output}")


if __name__ == "__main__":
    main()
