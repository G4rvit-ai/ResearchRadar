"""Application-facing facade for search and recommendations."""
import pandas as pd
from .keyword_extraction import extract_keywords
from .similarity_engine import ranked_search, similar_documents


class SearchEngine:
    def __init__(self, papers: pd.DataFrame, index, topic_model=None):
        self.papers, self.index, self.topic_model = papers.reset_index(drop=True), index, topic_model

    def search(self, query, top_k=10):
        results = []
        for row_index, score in ranked_search(self.index, query, top_k):
            paper = self.papers.iloc[row_index]
            results.append(self._result(row_index, paper, score))
        return results

    def recommendations(self, row_index, top_k=5):
        return [self._result(i, self.papers.iloc[i], score)
                for i, score in similar_documents(self.index, row_index, top_k)]

    def _result(self, row_index, paper, score):
        topic_id = None
        if self.topic_model is not None and row_index < len(self.topic_model.assignments):
            topic_id = int(self.topic_model.assignments[row_index])
        return {"row_index": row_index, "paper_id": paper.paper_id, "title": paper.title,
                "abstract": paper.abstract, "score": score,
                "keywords": [term for term, _ in extract_keywords(f"{paper.title}. {paper.abstract}", 8)],
                "topic_id": topic_id}
