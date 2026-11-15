# ResearchRadar

ResearchRadar is a reproducible Streamlit application for discovering machine-learning research papers with traditional NLP. It uses the Hugging Face `CShorten/ML-ArXiv-Papers` training split, validates title/abstract fields, removes empty and duplicate papers, and assigns stable content-based identifiers.

## Features

- Conservative scientific-text preprocessing that preserves terms such as BERT, CNN, RNN, LSTM, NLP, and Transformers.
- Sparse unigram/bigram TF-IDF indexing with configurable title boosting.
- Query and paper-to-paper cosine similarity without constructing a corpus-wide pairwise matrix.
- Configurable LDA topic modelling with keywords, counts, and paper exploration.
- Per-paper TF-IDF keyword extraction.
- Optional `MiniBatchKMeans` experiment in `src/clustering.py`.
- Streamlit pages for dashboard, search, recommendations, topics, keywords, analytics, and methodology.

## Setup and reproducible workflow

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_dataset.py --max-documents 1000  # bounded inspection sample
python scripts/build_models.py --max-documents 5000  # laptop smoke run
python scripts/build_clusters.py --clusters 10       # optional clustering experiment
streamlit run app.py
pytest -q
```

Use the full corpus for the final model build by omitting `--max-documents` from `build_models.py` only when intentionally running a long job. On an 8 GB MacBook Air, begin with 100–5,000 documents. A capped 10,000-paper configuration was validated on this project in about one minute with `--lda-documents 5000 --max-features 40000`; larger runs should be treated as experimental. The bounded commands use Hugging Face streaming and do not materialize all 117k rows. Model artifacts are stored under `models/`; the dataset is stored under `data/processed/`, and both are ignored by Git.

Configuration can be supplied through `RESEARCHRADAR_MAX_FEATURES`, `RESEARCHRADAR_LDA_DOCUMENTS`, `RESEARCHRADAR_N_TOPICS`, and `RESEARCHRADAR_TITLE_WEIGHT`, or by script flags such as `--max-features`, `--lda-documents`, `--topics`, and `--title-weight`. The default title weight is 2, represented by repeating normalized title text before vectorization. By default, LDA topic counts and paper lists describe its bounded training subset; TF-IDF search still covers every processed paper. Add `--assign-all-topics` when you intentionally want batched LDA assignments for every processed paper.

## Architecture

`dataset_loader.py` handles Hugging Face access and cleaning. `preprocessing.py` provides normalization and tokenization. `tfidf_engine.py` persists the sparse index and vectorizer. `similarity_engine.py` performs only one query/document sparse comparison at a time. `topic_modeling.py` limits LDA fitting to a configurable subset. `search_engine.py` is the UI-facing facade.

## Dataset, attribution, and limitations

Source: [CShorten/ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers), accessed with the Hugging Face `datasets` library. Review the dataset card and repository license before redistribution; this project does not redistribute the raw corpus. The documented fields do not reliably provide authors, dates, citations, or paper URLs, so none are fabricated. Search is primarily lexical TF-IDF retrieval, not a semantic embedding system. No search accuracy is claimed because the corpus contains no relevance labels.

## Demonstration queries

- `deep learning for medical image classification`
- `natural language processing in education`
- `sentiment analysis using machine learning`
- `neural networks for anomaly detection`
