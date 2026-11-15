"""Download the corpus and build reusable sparse search/topic artifacts."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SETTINGS, PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR
from src.dataset_loader import load_huggingface_dataset
from src.tfidf_engine import TfidfIndex
from src.topic_modeling import TopicModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-documents", type=int, default=SETTINGS.max_documents,
                        help="0 means all available papers; use 5000 for laptop development.")
    parser.add_argument("--lda-documents", type=int, default=SETTINGS.lda_documents)
    parser.add_argument("--topics", type=int, default=SETTINGS.n_topics)
    parser.add_argument("--max-features", type=int, default=SETTINGS.max_features)
    parser.add_argument("--title-weight", type=int, default=int(SETTINGS.title_weight))
    parser.add_argument("--assign-all-topics", action="store_true",
                        help="Assign LDA topics to all processed papers in batches.")
    args = parser.parse_args()
    papers = load_huggingface_dataset(SETTINGS.dataset_name, SETTINGS.split,
                                      max_documents=args.max_documents)
    papers.to_parquet(PROCESSED_DIR / "papers.parquet", index=False)
    index = TfidfIndex.fit(papers.title, papers.abstract, args.max_features, args.title_weight)
    index.save(MODELS_DIR / "tfidf")
    topic_model = TopicModel.fit(papers.title, papers.abstract, args.topics, args.lda_documents)
    if args.assign_all_topics:
        topic_model.assignments = topic_model.assign(papers.title.tolist(), papers.abstract.tolist(),
                                                     SETTINGS.batch_size)
    topic_model.save(MODELS_DIR / "lda")
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps({
        "dataset": SETTINGS.dataset_name, "split": SETTINGS.split,
        "paper_count": len(papers), "topic_count": args.topics,
        "vector_shape": list(index.matrix.shape),
    }, indent=2), encoding="utf-8")
    print(f"Built artifacts for {len(papers):,} papers at {MODELS_DIR}")


if __name__ == "__main__":
    main()
