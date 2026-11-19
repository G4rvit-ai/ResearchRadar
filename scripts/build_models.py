"""Download the corpus and build reusable sparse search/topic artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import shutil
import sys
import pandas as pd
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, SETTINGS, PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR
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
    if args.max_documents < 0:
        parser.error("--max-documents must be 0 or a positive integer")
    if args.lda_documents < 0:
        parser.error("--lda-documents must be 0 or a positive integer")
    if args.topics < 1:
        parser.error("--topics must be at least 1")
    if args.max_features < 1:
        parser.error("--max-features must be a positive integer")
    if args.title_weight < 1:
        parser.error("--title-weight must be a positive integer")
    papers = load_huggingface_dataset(SETTINGS.dataset_name, SETTINGS.split,
                                      max_documents=args.max_documents)
    corpus_hasher = hashlib.sha256()
    for paper_id, title, abstract in papers[["paper_id", "title", "abstract"]].itertuples(index=False, name=None):
        corpus_hasher.update(f"{paper_id}\0{title}\0{abstract}\0".encode("utf-8"))
    corpus_fingerprint = corpus_hasher.hexdigest()
    index = TfidfIndex.fit(papers.title, papers.abstract, args.max_features, args.title_weight)
    topic_model = TopicModel.fit(papers.title, papers.abstract, args.topics, args.lda_documents,
                                 args.max_features)
    if args.assign_all_topics:
        topic_model.assignments = topic_model.assign(papers.title.tolist(), papers.abstract.tolist(),
                                                     SETTINGS.batch_size)
    metadata = {
        "dataset": SETTINGS.dataset_name, "split": SETTINGS.split,
        "paper_count": len(papers), "topic_count": args.topics,
        "vector_shape": list(index.matrix.shape),
        "corpus_fingerprint": corpus_fingerprint,
        "lda_documents": args.lda_documents,
        "max_features": args.max_features,
        "title_weight": args.title_weight,
    }
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        temporary = Path(temporary)
        staged_papers = temporary / "papers.parquet"
        staged_tfidf = temporary / "tfidf"
        staged_lda = temporary / "lda"
        staged_metadata = temporary / "metadata.json"
        papers.to_parquet(staged_papers, index=False)
        index.save(staged_tfidf)
        topic_model.save(staged_lda)
        staged_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        for target in (MODELS_DIR / "tfidf", MODELS_DIR / "lda"):
            if target.exists():
                shutil.rmtree(target)
        os.replace(staged_papers, PROCESSED_DIR / "papers.parquet")
        os.replace(staged_tfidf, MODELS_DIR / "tfidf")
        os.replace(staged_lda, MODELS_DIR / "lda")
        os.replace(staged_metadata, ARTIFACTS_DIR / "metadata.json")
    print(f"Built artifacts for {len(papers):,} papers at {MODELS_DIR}")


if __name__ == "__main__":
    main()
