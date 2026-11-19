import numpy as np
import pandas as pd
import subprocess
import sys
from src.dataset_loader import clean_dataframe
from src.preprocessing import preprocess_text, combine_title_abstract
from src.tfidf_engine import TfidfIndex
from src.similarity_engine import ranked_search, similar_documents
from src.keyword_extraction import extract_keywords
from src.topic_modeling import TopicModel
from src.search_engine import SearchEngine
from src.clustering import fit_clusters
from src.dataset_loader import load_huggingface_dataset


TITLES = ["Deep learning for images", "Natural language models", "Deep neural image networks", "Language classification"]
ABSTRACTS = ["cnn image classification with neural networks", "transformer language representation", "neural networks classify medical images", "language model sentiment classification"]


def test_cleaning_missing_duplicates():
    frame = pd.DataFrame({"Unnamed: 0": [1, 2, 3], "title": ["A", "A", None], "abstract": ["B", "B", "C"]})
    result = clean_dataframe(frame)
    assert len(result) == 1 and list(result.columns) == ["paper_id", "title", "abstract"]


def test_preprocessing_preserves_technical_terms():
    text = preprocess_text("BERT and CNN for NLP")
    assert "bert" in text and "cnn" in text and "nlp" in text
    assert " and " not in f" {text} "
    assert combine_title_abstract("BERT", "model").count("bert") == 2


def test_search_and_recommendations_are_ranked_and_exclude_self():
    index = TfidfIndex.fit(TITLES, ABSTRACTS, max_features=1000)
    results = ranked_search(index, "deep image networks", 2)
    assert results[0][0] in (0, 2) and results[0][1] >= results[1][1]
    recommendations = similar_documents(index, results[0][0], 3)
    assert all(row != results[0][0] for row, _ in recommendations)
    assert ranked_search(index, "quantum cryptography", 3) == []


def test_keywords_and_topics():
    assert extract_keywords("transformer transformer attention neural model", 2)[0][0] == "transformer"
    assert extract_keywords("the and or", 5) == []
    topic = TopicModel.fit(TITLES, ABSTRACTS, n_topics=2, max_documents=4, max_features=100)
    assert len(topic.topic_summary()) == 2
    assert np.all((topic.assignments >= 0) & (topic.assignments < 2))
    assignments = topic.assign(TITLES, ABSTRACTS, batch_size=2)
    assert len(assignments) == len(TITLES)


def test_tfidf_persistence_and_search_facade(tmp_path):
    papers = pd.DataFrame({"paper_id": ["1", "2", "3", "4"],
                           "title": TITLES, "abstract": ABSTRACTS})
    index = TfidfIndex.fit(TITLES, ABSTRACTS, max_features=1000)
    index.save(tmp_path / "tfidf")
    restored = TfidfIndex.load(tmp_path / "tfidf")
    engine = SearchEngine(papers, restored)
    assert engine.search("deep image", 1)[0]["paper_id"] in {"1", "3"}


def test_model_configuration_edge_cases():
    tiny_index = TfidfIndex.fit(["C++"], ["C++ code"], max_features=100)
    assert tiny_index.matrix.shape[0] == 1
    assert "c++" in tiny_index.vectorizer.vocabulary_
    topic = TopicModel.fit(TITLES, ABSTRACTS, n_topics=20, max_documents=0, max_features=100)
    assert len(topic.topic_summary()) == len(TITLES)
    try:
        TfidfIndex.fit([], [], max_features=100)
    except ValueError as exc:
        assert "At least one document" in str(exc)
    else:
        raise AssertionError("Empty TF-IDF input must fail explicitly.")
    try:
        TopicModel.fit(TITLES, ABSTRACTS, n_topics=2, max_features=100).assign(
            TITLES, ABSTRACTS, batch_size=0
        )
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("Invalid topic batch size must fail explicitly.")


def test_optional_clustering():
    index = TfidfIndex.fit(TITLES, ABSTRACTS, max_features=1000)
    model, labels = fit_clusters(index.matrix, n_clusters=2)
    assert model.n_clusters == 2
    assert len(labels) == len(TITLES)


def test_negative_limits_and_fractional_title_weight():
    try:
        load_huggingface_dataset(max_documents=-1)
    except ValueError as exc:
        assert "0 or a positive" in str(exc)
    else:
        raise AssertionError("Negative document limits must fail explicitly.")
    try:
        TfidfIndex.fit(TITLES, ABSTRACTS, title_weight=1.5)
    except ValueError as exc:
        assert "positive integer" in str(exc)
    else:
        raise AssertionError("Fractional title weights must fail explicitly.")


def test_script_rejects_negative_document_limit():
    result = subprocess.run(
        [sys.executable, "scripts/build_models.py", "--max-documents", "-1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must be 0 or a positive integer" in result.stderr
