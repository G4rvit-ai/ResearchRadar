"""Conservative preprocessing helpers for scientific text."""
import re
from typing import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.-]{1,}")
_STOPWORDS = set(ENGLISH_STOP_WORDS) - {
    "above", "below", "between", "both", "down", "few", "more", "most",
    "over", "same", "under", "up", "very",
}


def normalize_text(text: object) -> str:
    """Return stable searchable text while preserving technical token characters."""
    if text is None:
        return ""
    value = str(text).replace("\u0000", " ").lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokenize(text: object) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def preprocess_text(text: object) -> str:
    tokens = [token for token in tokenize(text) if token not in _STOPWORDS and len(token) > 1]
    return " ".join(tokens)


def combine_title_abstract(title: object, abstract: object, title_weight: int = 2) -> str:
    title_text = preprocess_text(title)
    abstract_text = preprocess_text(abstract)
    return " ".join([title_text] * max(1, title_weight) + [abstract_text])


def meaningful_terms(texts: Iterable[str], limit: int = 30) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for text in texts:
        for token in set(tokenize(text)):
            if token not in _STOPWORDS and len(token) > 2:
                counts[token] = counts.get(token, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
