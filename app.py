"""ResearchRadar Streamlit dashboard."""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import ARTIFACTS_DIR, MODELS_DIR, PROCESSED_DIR
from src.search_engine import SearchEngine
from src.tfidf_engine import TfidfIndex
from src.topic_modeling import TopicModel
from src.preprocessing import meaningful_terms


st.set_page_config(page_title="ResearchRadar", page_icon="📚", layout="wide")


@st.cache_resource
def load_engine():
    papers_path = PROCESSED_DIR / "papers.parquet"
    if not papers_path.exists() or not (MODELS_DIR / "tfidf" / "vectorizer.joblib").exists():
        return None
    papers = pd.read_parquet(papers_path)
    index = TfidfIndex.load(MODELS_DIR / "tfidf")
    topic_model = TopicModel.load(MODELS_DIR / "lda") if (MODELS_DIR / "lda" / "topic_model.joblib").exists() else None
    return SearchEngine(papers, index, topic_model)


def render_result(result, engine):
    with st.container(border=True):
        st.subheader(result["title"])
        st.caption(f"Relevance: {result['score']:.3f}" + (f" · Topic {result['topic_id']}" if result["topic_id"] is not None else ""))
        st.write(result["abstract"][:500] + ("..." if len(result["abstract"]) > 500 else ""))
        st.write("**Keywords:** " + ", ".join(result["keywords"]))
        with st.expander("Full abstract and similar papers"):
            st.write(result["abstract"])
            st.markdown("**Recommendations**")
            for recommendation in engine.recommendations(result["row_index"], 5):
                st.write(f"- {recommendation['title']} ({recommendation['score']:.3f})")


def main():
    st.title("📚 ResearchRadar")
    st.caption("Research paper discovery and topic intelligence using traditional NLP.")
    engine = load_engine()
    if engine is None:
        st.warning("No local model artifacts found. Build them first with `python scripts/build_models.py --max-documents 5000`.")
        st.code("python -m pip install -r requirements.txt\npython scripts/build_models.py --max-documents 5000\nstreamlit run app.py")
        return
    papers = engine.papers
    topics = engine.topic_model.topic_summary() if engine.topic_model else []
    page = st.sidebar.radio("Explore", ["Home", "Research Search", "Topic Explorer", "Keyword Explorer", "Research Analytics", "About"])
    if page == "Home":
        st.header("Explore the machine-learning literature")
        a, b, c = st.columns(3)
        a.metric("Processed papers", f"{len(papers):,}")
        b.metric("Discovered topics", len(topics))
        c.metric("TF-IDF features", f"{engine.index.matrix.shape[1]:,}")
        st.info("Search is lexical: TF-IDF gives more weight to title terms and cosine similarity ranks matching papers.")
        if topics:
            st.plotly_chart(px.bar(pd.DataFrame(topics), x="topic_id", y="paper_count", title="Papers assigned to each LDA topic"), use_container_width=True)
        st.subheader("Common research terms")
        st.write(", ".join(term for term, _ in meaningful_terms(papers.abstract.head(10000), 25)))
    elif page == "Research Search":
        st.header("Research Paper Search")
        query = st.text_input("Describe a research question or enter keywords", placeholder="deep learning for medical image classification")
        top_k = st.selectbox("Results", [5, 10, 20], index=1)
        if query:
            results = engine.search(query, top_k)
            st.write(f"Found {len(results)} ranked results.")
            for result in results:
                render_result(result, engine)
    elif page == "Topic Explorer":
        st.header("LDA Topic Explorer")
        if not topics:
            st.info("Topic artifacts were not generated.")
        else:
            selected = st.selectbox("Topic", range(len(topics)), format_func=lambda i: f"Topic {i}")
            topic = topics[selected]
            st.write("**Keywords:** " + ", ".join(topic["keywords"]))
            st.metric("Papers in topic", topic["paper_count"])
            rows = engine.papers.iloc[:len(engine.topic_model.assignments)]
            topic_rows = rows[engine.topic_model.assignments == selected]
            for _, paper in topic_rows.head(10).iterrows():
                st.write(f"- {paper.title}")
    elif page == "Keyword Explorer":
        st.header("Keyword Explorer")
        keyword = st.text_input("Keyword", placeholder="transformer")
        if keyword:
            results = engine.search(keyword, 10)
            st.write(f"Top papers associated with **{keyword}**")
            for result in results:
                render_result(result, engine)
    elif page == "Research Analytics":
        st.header("Research Analytics")
        if topics:
            frame = pd.DataFrame(topics)
            st.plotly_chart(px.bar(frame, x="topic_id", y="paper_count", color="topic_id", title="Topic distribution"), use_container_width=True)
            st.dataframe(frame, use_container_width=True)
    else:
        st.header("About ResearchRadar")
        st.markdown("""ResearchRadar demonstrates information retrieval with a reproducible academic corpus.

**TF-IDF** represents how important a term is to one document relative to the corpus. **Cosine similarity** compares the direction of sparse document vectors. **LDA** discovers probabilistic groups of co-occurring terms. Keywords are extracted from per-paper TF-IDF weights.

The source is the [ML-ArXiv-Papers dataset](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers). The documented fields are title and abstract, so this app intentionally does not invent authors, dates, citations, or paper links. Search is lexical and may miss semantically related wording.""")


if __name__ == "__main__":
    main()
