"""Hugging Face dataset loading, validation, and deterministic cleaning."""
from pathlib import Path
import hashlib
import pandas as pd


def load_huggingface_dataset(dataset_name: str = "CShorten/ML-ArXiv-Papers", split: str = "train",
                             cache_dir: str | Path | None = None,
                             max_documents: int = 0) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package before downloading the corpus.") from exc
    if max_documents > 0:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir, streaming=True)
        rows = []
        for row in dataset:
            rows.append(row)
            if len(rows) >= max_documents:
                break
        frame = pd.DataFrame(rows)
    else:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        frame = dataset.to_pandas()
    return clean_dataframe(frame)


def clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate documented title/abstract fields and preserve original display text."""
    result = frame.copy()
    result = result.drop(columns=[column for column in result.columns if column.lower().startswith("unnamed")],
                         errors="ignore")
    title_col = next((c for c in result.columns if c.lower() == "title"), None)
    abstract_col = next((c for c in result.columns if c.lower() in {"abstract", "summary"}), None)
    if title_col is None or abstract_col is None:
        raise ValueError(f"Dataset must contain title and abstract/summary columns; found {list(result.columns)}")
    result = result.rename(columns={title_col: "title", abstract_col: "abstract"})
    result["title"] = result["title"].fillna("").astype(str).str.strip()
    result["abstract"] = result["abstract"].fillna("").astype(str).str.strip()
    result = result[(result["title"] != "") & (result["abstract"] != "")]
    result["_dedupe_key"] = (result["title"].str.lower().str.replace(r"\W+", "", regex=True)
                             + "|" + result["abstract"].str.lower().str.replace(r"\W+", "", regex=True))
    result = result.drop_duplicates("_dedupe_key").drop(columns="_dedupe_key").reset_index(drop=True)
    result.insert(0, "paper_id", [hashlib.sha1(f"{title}\n{abstract}".encode("utf-8")).hexdigest()[:16]
                                  for title, abstract in zip(result["title"], result["abstract"])])
    return result[["paper_id", "title", "abstract"]]
