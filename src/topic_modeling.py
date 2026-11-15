"""Memory-conscious LDA topic fitting and assignment."""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .preprocessing import TOKEN_PATTERN, combine_title_abstract


class TopicModel:
    def __init__(self, vectorizer, model, assignments=None):
        self.vectorizer, self.model, self.assignments = vectorizer, model, assignments

    @classmethod
    def fit(cls, titles, abstracts, n_topics=10, max_documents=20000, max_features=30000):
        documents = [combine_title_abstract(t, a) for t, a in zip(titles, abstracts)]
        if not documents:
            raise ValueError("At least one document is required to fit the topic model.")
        if n_topics < 1:
            raise ValueError("n_topics must be at least 1.")
        if max_documents > 0:
            documents = documents[:max_documents]
        n_topics = min(n_topics, len(documents))
        min_df = max(1, min(3, len(documents) // 10))
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=min_df,
                                     token_pattern=TOKEN_PATTERN,
                                     max_features=max_features)
        counts = vectorizer.fit_transform(documents)
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                          learning_method="batch", max_iter=10, n_jobs=1)
        model.fit(counts)
        assignments = model.transform(counts).argmax(axis=1)
        return cls(vectorizer, model, assignments)

    def topic_keywords(self, topic_id: int, limit=12):
        terms = self.vectorizer.get_feature_names_out()
        order = self.model.components_[topic_id].argsort()[::-1][:limit]
        return [(str(terms[i]), float(self.model.components_[topic_id, i])) for i in order]

    def assign(self, titles, abstracts, batch_size=2048):
        """Assign documents in batches after fitting, avoiding a dense corpus matrix."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        assignments = []
        for start in range(0, len(titles), batch_size):
            documents = [
                combine_title_abstract(title, abstract)
                for title, abstract in zip(titles[start:start + batch_size],
                                           abstracts[start:start + batch_size])
            ]
            counts = self.vectorizer.transform(documents)
            assignments.append(self.model.transform(counts).argmax(axis=1))
        return np.concatenate(assignments) if assignments else np.array([], dtype=int)

    def topic_summary(self):
        if self.assignments is None:
            raise RuntimeError("Topic assignments are unavailable for this model artifact.")
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
