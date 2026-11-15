"""Sparse TF-IDF indexing with title boosting."""
from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import TOKEN_PATTERN, combine_title_abstract


class TfidfIndex:
    def __init__(self, vectorizer: TfidfVectorizer, matrix: csr_matrix, title_weight: int = 2):
        self.vectorizer, self.matrix, self.title_weight = vectorizer, matrix, title_weight

    @classmethod
    def fit(cls, titles, abstracts, max_features=100000, title_weight=2):
        documents = [combine_title_abstract(title, abstract, title_weight) for title, abstract in zip(titles, abstracts)]
        if not documents:
            raise ValueError("At least one document is required to fit the TF-IDF index.")
        document_count = len(documents)
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=min(2, document_count),
            max_df=0.98 if document_count > 2 else 1.0,
            token_pattern=TOKEN_PATTERN,
                                     max_features=max_features, sublinear_tf=True)
        return cls(vectorizer, vectorizer.fit_transform(documents).tocsr(), title_weight)

    def transform_query(self, query: str):
        return self.vectorizer.transform([combine_title_abstract(query, "", self.title_weight)])

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "title_weight": self.title_weight}, directory / "vectorizer.joblib")
        save_npz(directory / "tfidf.npz", self.matrix)

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)
        payload = joblib.load(directory / "vectorizer.joblib")
        return cls(payload["vectorizer"], load_npz(directory / "tfidf.npz").tocsr(), payload["title_weight"])
