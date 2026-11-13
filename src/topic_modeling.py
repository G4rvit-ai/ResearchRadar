"""Memory-conscious LDA topic fitting and assignment."""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .preprocessing import combine_title_abstract


class TopicModel:
    def __init__(self, vectorizer, model, assignments=None):
        self.vectorizer, self.model, self.assignments = vectorizer, model, assignments

    @classmethod
    def fit(cls, titles, abstracts, n_topics=10, max_documents=20000, max_features=30000):
        documents = [combine_title_abstract(t, a) for t, a in zip(titles, abstracts)]
        documents = documents[:max_documents]
        min_df = max(1, min(3, len(documents) // 10))
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=min_df,
                                     max_features=max_features)
        counts = vectorizer.fit_transform(documents)
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                          learning_method="batch", max_iter=10, n_jobs=-1)
        model.fit(counts)
        assignments = model.transform(counts).argmax(axis=1)
        return cls(vectorizer, model, assignments)

    def topic_keywords(self, topic_id: int, limit=12):
        terms = self.vectorizer.get_feature_names_out()
        order = self.model.components_[topic_id].argsort()[::-1][:limit]
        return [(str(terms[i]), float(self.model.components_[topic_id, i])) for i in order]

    def topic_summary(self):
        return [{"topic_id": i, "keywords": [term for term, _ in self.topic_keywords(i)],
                 "paper_count": int((self.assignments == i).sum())}
                for i in range(self.model.n_components)]

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "topic_model.joblib")
        (directory / "topics.json").write_text(json.dumps(self.topic_summary(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path):
        return joblib.load(Path(directory) / "topic_model.joblib")
