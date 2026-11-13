import numpy as np
import pandas as pd
from src.dataset_loader import clean_dataframe
from src.preprocessing import preprocess_text, combine_title_abstract
from src.tfidf_engine import TfidfIndex
from src.similarity_engine import ranked_search, similar_documents
from src.keyword_extraction import extract_keywords
from src.topic_modeling import TopicModel


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


def test_keywords_and_topics():
    assert extract_keywords("transformer transformer attention neural model", 2)[0][0] == "transformer"
    topic = TopicModel.fit(TITLES, ABSTRACTS, n_topics=2, max_documents=4, max_features=100)
    assert len(topic.topic_summary()) == 2
    assert np.all((topic.assignments >= 0) & (topic.assignments < 2))
