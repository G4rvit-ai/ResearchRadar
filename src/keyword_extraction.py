"""TF-IDF keyword extraction for individual papers."""
from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocessing import TOKEN_PATTERN


def extract_keywords(text: str, limit: int = 10) -> list[tuple[str, float]]:
    if not text or not text.strip():
        return []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                                 token_pattern=TOKEN_PATTERN, max_features=1000)
    try:
        matrix = vectorizer.fit_transform([text])
    except ValueError as exc:
        if "empty vocabulary" in str(exc):
            return []
        raise
    scores = matrix.toarray()[0]
    terms = vectorizer.get_feature_names_out()
    order = scores.argsort()[::-1]
    return [(str(terms[i]), float(scores[i])) for i in order[:limit] if scores[i] > 0]
