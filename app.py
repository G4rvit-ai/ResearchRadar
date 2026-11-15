"""ResearchRadar Streamlit dashboard."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import ARTIFACTS_DIR, MODELS_DIR, PROCESSED_DIR
from src.search_engine import SearchEngine
from src.tfidf_engine import TfidfIndex
from src.topic_modeling import TopicModel
from src.preprocessing import meaningful_terms


st.set_page_config(page_title="ResearchRadar", page_icon="📚", layout="wide")


def style_chart(figure):
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": "#aab8d0"},
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        title={"font": {"family": "Space Grotesk, sans-serif", "size": 17, "color": "#e8eefc"}},
    )
    return figure


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --rr-ink: #e8eefc;
            --rr-muted: #9aa8c3;
            --rr-panel: #111a2b;
            --rr-panel-2: #17243a;
            --rr-border: rgba(148, 163, 184, .17);
            --rr-blue: #79b8ff;
            --rr-purple: #a78bfa;
            --rr-teal: #54d6c7;
        }

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }
        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -.03em;
        }
        .block-container {
            max-width: 1440px;
            padding: 2rem 4rem 4rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0c1424 0%, #101a2d 100%);
            border-right: 1px solid var(--rr-border);
        }
        [data-testid="stSidebar"] .block-container {
            padding: 2rem 1.25rem;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(28, 43, 69, .95), rgba(17, 26, 43, .95));
            border: 1px solid var(--rr-border);
            border-radius: 16px;
            padding: 1rem 1.15rem;
            box-shadow: 0 12px 30px rgba(0, 0, 0, .14);
        }
        [data-testid="stMetricLabel"] {
            color: var(--rr-muted);
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: var(--rr-ink);
            font-family: 'Space Grotesk', sans-serif;
        }
        .rr-brand {
            display: flex;
            align-items: center;
            gap: .7rem;
            margin-bottom: 2.2rem;
        }
        .rr-brand-mark {
            display: grid;
            place-items: center;
            width: 42px;
            height: 42px;
            border-radius: 13px;
            background: linear-gradient(135deg, #72b7ff, #8b5cf6);
            font-size: 1.35rem;
            box-shadow: 0 8px 24px rgba(99, 102, 241, .35);
        }
        .rr-brand-name {
            font: 700 1.1rem 'Space Grotesk', sans-serif;
            color: var(--rr-ink);
        }
        .rr-brand-subtitle {
            color: var(--rr-muted);
            font-size: .72rem;
            margin-top: .12rem;
        }
        .rr-hero {
            position: relative;
            overflow: hidden;
            padding: 2.8rem 3rem;
            margin: .4rem 0 2rem;
            border: 1px solid rgba(121, 184, 255, .2);
            border-radius: 24px;
            background:
                radial-gradient(circle at 85% 15%, rgba(167, 139, 250, .28), transparent 30%),
                radial-gradient(circle at 55% 110%, rgba(84, 214, 199, .17), transparent 30%),
                linear-gradient(120deg, #132746 0%, #18213d 58%, #222047 100%);
            box-shadow: 0 24px 60px rgba(0, 0, 0, .2);
        }
        .rr-eyebrow {
            color: var(--rr-teal);
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: .8rem;
        }
        .rr-hero h1 {
            color: #f3f7ff;
            font-size: clamp(2.2rem, 5vw, 4.3rem);
            line-height: 1.02;
            margin: 0;
            max-width: 760px;
        }
        .rr-hero p {
            color: #bdcbe2;
            font-size: 1.04rem;
            line-height: 1.7;
            max-width: 680px;
            margin: 1.15rem 0 0;
        }
        .rr-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: .55rem;
            margin-top: 1.45rem;
        }
        .rr-chip {
            padding: .43rem .75rem;
            color: #dce8fb;
            background: rgba(255, 255, 255, .08);
            border: 1px solid rgba(255, 255, 255, .12);
            border-radius: 999px;
            font-size: .78rem;
        }
        .rr-section-kicker {
            color: var(--rr-blue);
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin: 1.8rem 0 .45rem;
        }
        .rr-section-copy {
            color: var(--rr-muted);
            margin-bottom: 1.1rem;
        }
        .rr-card {
            height: 100%;
            padding: 1.2rem 1.25rem;
            border: 1px solid var(--rr-border);
            border-radius: 17px;
            background: linear-gradient(145deg, rgba(24, 37, 61, .92), rgba(15, 24, 40, .92));
        }
        .rr-card-icon {
            font-size: 1.35rem;
            margin-bottom: .65rem;
        }
        .rr-card-title {
            color: var(--rr-ink);
            font: 600 1rem 'Space Grotesk', sans-serif;
            margin-bottom: .4rem;
        }
        .rr-card-copy {
            color: var(--rr-muted);
            font-size: .86rem;
            line-height: 1.55;
        }
        .rr-result-title {
            color: #e8eefc;
            font: 600 1.05rem 'Space Grotesk', sans-serif;
            line-height: 1.35;
        }
        .rr-result-meta {
            color: var(--rr-muted);
            font-size: .8rem;
            margin: .4rem 0 .8rem;
        }
        .rr-score {
            color: var(--rr-teal);
            font-weight: 700;
        }
        .stButton > button {
            border-radius: 10px;
            border: 1px solid rgba(121, 184, 255, .28);
        }
        div[data-testid="stExpander"] {
            border-color: var(--rr-border);
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_engine():
    papers_path = PROCESSED_DIR / "papers.parquet"
    if not papers_path.exists() or not (MODELS_DIR / "tfidf" / "vectorizer.joblib").exists():
        return None
    papers = pd.read_parquet(papers_path)
    index = TfidfIndex.load(MODELS_DIR / "tfidf")
    if len(papers) != index.matrix.shape[0]:
        raise RuntimeError(
            "Processed papers and TF-IDF index contain different numbers of rows. "
            "Rebuild artifacts with scripts/build_models.py."
        )
    topic_model = TopicModel.load(MODELS_DIR / "lda") if (MODELS_DIR / "lda" / "topic_model.joblib").exists() else None
    if topic_model is not None and len(topic_model.assignments) > len(papers):
        raise RuntimeError(
            "The LDA artifact contains more assignments than the processed papers. "
            "Rebuild artifacts with scripts/build_models.py."
        )
    return SearchEngine(papers, index, topic_model)


@st.cache_data
def load_cluster_labels():
    cluster_path = MODELS_DIR / "clustering.joblib"
    if not cluster_path.exists():
        return None
    import joblib
    return joblib.load(cluster_path)["labels"]


def result_for_row(engine, row_index):
    """Build a paper result while tolerating a resource cached by an older app version."""
    if hasattr(engine, "result_for_row"):
        return engine.result_for_row(row_index)
    paper = engine.papers.iloc[row_index]
    return engine._result(row_index, paper, 0.0)


def render_result(result, engine):
    with st.container(border=True):
        saved = result["paper_id"] in st.session_state["saved_papers"]
        title_col, action_col = st.columns([8, 1])
        with action_col:
            if st.button("★" if saved else "☆", key=f"save-{result['paper_id']}",
                         help="Remove from saved papers" if saved else "Save this paper"):
                if saved:
                    st.session_state["saved_papers"].discard(result["paper_id"])
                else:
                    st.session_state["saved_papers"].add(result["paper_id"])
                st.rerun()
        topic_label = f" · Topic {result['topic_id']}" if result["topic_id"] is not None else ""
        with title_col:
            st.markdown(
                f"<div class='rr-result-title'>{result['title']}</div>"
                f"<div class='rr-result-meta'><span class='rr-score'>● {result['score']:.3f} relevance</span>{topic_label}</div>",
                unsafe_allow_html=True,
            )
        st.write(result["abstract"][:500] + ("..." if len(result["abstract"]) > 500 else ""))
        st.write("**Keywords:** " + ", ".join(result["keywords"]))
        with st.expander("Full abstract and similar papers"):
            st.write(result["abstract"])
            st.markdown("**Recommendations**")
            for recommendation in engine.recommendations(result["row_index"], 5):
                st.write(f"- {recommendation['title']} ({recommendation['score']:.3f})")


def main():
    inject_styles()
    st.session_state.setdefault("saved_papers", set())
    engine = load_engine()
    if engine is None:
        st.warning("No local model artifacts found. Build them first with `python scripts/build_models.py --max-documents 5000`.")
        st.code("python -m pip install -r requirements.txt\npython scripts/build_models.py --max-documents 5000\nstreamlit run app.py")
        return
    papers = engine.papers
    topics = engine.topic_model.topic_summary() if engine.topic_model else []
    st.sidebar.markdown(
        """
        <div class="rr-brand">
            <div class="rr-brand-mark">📚</div>
            <div><div class="rr-brand-name">ResearchRadar</div>
            <div class="rr-brand-subtitle">Topic intelligence platform</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.markdown("**CORPUS**")
    st.sidebar.caption("ML-ArXiv-Papers · Hugging Face")
    st.sidebar.caption(f"{len(papers):,} papers indexed")
    st.sidebar.caption(f"{engine.index.matrix.shape[1]:,} searchable features")
    page = st.sidebar.radio(
        "WORKSPACE",
        ["⌂  Home", "⌕  Research Search", "◈  Topic Explorer",
         "✦  Keyword Explorer", "▦  Research Analytics",
         f"★  Saved Papers ({len(st.session_state['saved_papers'])})", "ⓘ  About"],
        label_visibility="visible",
    )
    page = page.split("  ", 1)[1]
    if page == "Home":
        st.markdown(
            """
            <div class="rr-hero">
                <div class="rr-eyebrow">Research discovery · powered by NLP</div>
                <h1>Find the ideas<br>behind the papers.</h1>
                <p>ResearchRadar helps you navigate machine-learning literature with transparent,
                traditional NLP methods you can understand, inspect, and reproduce.</p>
                <div class="rr-chip-row">
                    <span class="rr-chip">TF-IDF retrieval</span>
                    <span class="rr-chip">Cosine similarity</span>
                    <span class="rr-chip">LDA topic modelling</span>
                    <span class="rr-chip">Keyword intelligence</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='rr-section-kicker'>Corpus overview</div>", unsafe_allow_html=True)
        st.markdown("<div class='rr-section-copy'>A live view of the processed ML-ArXiv research collection.</div>", unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric("Processed papers", f"{len(papers):,}")
        b.metric("Discovered topics", len(topics))
        c.metric("TF-IDF features", f"{engine.index.matrix.shape[1]:,}")
        st.markdown(
            """
            <div class="rr-card" style="margin-top: 1.2rem;">
                <div class="rr-card-icon">⚡</div>
                <div class="rr-card-title">A transparent research workflow</div>
                <div class="rr-card-copy">Enter a question, retrieve papers with title-boosted TF-IDF,
                inspect the relevance score, then follow the topic and similarity trails.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='rr-section-kicker'>How it works</div>", unsafe_allow_html=True)
        workflow = st.columns(3)
        for column, icon, title, copy in zip(
            workflow,
            ["01", "02", "03"],
            ["Search", "Understand", "Explore"],
            [
                "Your query is normalized and compared against sparse paper representations.",
                "LDA surfaces recurring topic vocabulary across the research collection.",
                "Open a paper to discover keyword signals and cosine-similar publications.",
            ],
        ):
            with column:
                st.markdown(
                    f"<div class='rr-card'><div class='rr-card-icon'>{icon}</div>"
                    f"<div class='rr-card-title'>{title}</div><div class='rr-card-copy'>{copy}</div></div>",
                    unsafe_allow_html=True,
                )
        if topics:
            st.plotly_chart(style_chart(px.bar(pd.DataFrame(topics), x="topic_id", y="paper_count",
                                               title="Papers assigned to each LDA topic")),
                            use_container_width=True)
            if len(engine.topic_model.assignments) < len(papers):
                st.caption(
                    f"Topic counts are based on the {len(engine.topic_model.assignments):,}-paper "
                    "LDA training subset; search covers all processed papers."
                )
        st.markdown("<div class='rr-section-kicker'>Corpus signals</div>", unsafe_allow_html=True)
        st.subheader("Common research terms")
        common_terms = meaningful_terms(papers.abstract.head(10000), 25)
        st.write(", ".join(term for term, _ in common_terms))
        st.plotly_chart(
            style_chart(px.bar(pd.DataFrame(common_terms, columns=["term", "document_count"]).head(15),
                               x="document_count", y="term", orientation="h",
                               title="Common terms in paper abstracts")),
            use_container_width=True,
        )
    elif page == "Research Search":
        st.header("Research Paper Search")
        st.markdown("<div class='rr-section-copy'>Search by research question, method, application area, or technical concept.</div>", unsafe_allow_html=True)
        query = st.text_input("Describe a research question or enter keywords", placeholder="e.g. deep learning for medical image classification", label_visibility="collapsed")
        controls = st.columns([2, 1])
        top_k = controls[0].selectbox("Results", [5, 10, 20], index=1)
        min_score = controls[1].slider("Minimum relevance", 0.0, 1.0, 0.0, 0.05)
        if query:
            results = engine.search(query, top_k)
            results = [result for result in results if result["score"] >= min_score]
            st.write(f"Found {len(results)} ranked results.")
            if not results:
                st.info("No lexical matches were found. Try broader terms or a related technical phrase.")
            else:
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
            if len(engine.topic_model.assignments) < len(papers):
                st.caption("This topic's paper list is limited to the LDA training subset.")
            keyword_frame = pd.DataFrame(engine.topic_model.topic_keywords(selected, 10),
                                         columns=["term", "weight"])
            st.plotly_chart(style_chart(px.bar(keyword_frame.sort_values("weight"), x="weight", y="term",
                                               orientation="h", title=f"Topic {selected} keywords")),
                            use_container_width=True)
            rows = engine.papers.iloc[:len(engine.topic_model.assignments)]
            topic_rows = rows[engine.topic_model.assignments == selected]
            st.markdown("#### Representative papers")
            for row_index, paper in topic_rows.head(10).iterrows():
                with st.expander(paper.title):
                    st.write(paper.abstract)
                    result = result_for_row(engine, row_index)
                    st.write("**Keywords:** " + ", ".join(result["keywords"]))
    elif page == "Keyword Explorer":
        st.header("Keyword Explorer")
        keyword = st.text_input("Keyword", placeholder="transformer")
        if keyword:
            results = engine.search(keyword, 10)
            st.write(f"Top papers associated with **{keyword}**")
            if results:
                term_frame = pd.DataFrame(
                    [(term, score) for result in results for term in result["keywords"] if keyword.lower() in term.lower()],
                    columns=["term", "papers"],
                ).groupby("term", as_index=False).size().rename(columns={"size": "papers"})
                if not term_frame.empty:
                    st.plotly_chart(style_chart(px.bar(term_frame.sort_values("papers"),
                                                       x="papers", y="term", orientation="h",
                                                       title="Keyword associations in top results")),
                                    use_container_width=True)
                for result in results:
                    render_result(result, engine)
            else:
                st.info("No papers matched that keyword.")
    elif page == "Research Analytics":
        st.header("Research Analytics")
        if topics:
            frame = pd.DataFrame(topics)
            st.plotly_chart(style_chart(px.bar(frame, x="topic_id", y="paper_count", color="topic_id",
                                               title="Topic distribution")),
                            use_container_width=True)
            st.dataframe(frame, use_container_width=True)
        common_terms = meaningful_terms(papers.abstract.head(10000), 20)
        st.plotly_chart(style_chart(px.bar(pd.DataFrame(common_terms, columns=["term", "document_count"]),
                                           x="document_count", y="term", orientation="h",
                                           title="Most frequent meaningful terms")),
                        use_container_width=True)
        labels = load_cluster_labels()
        if labels is not None and len(labels) != len(papers):
            st.warning("Clustering artifact is stale. Rebuild it with `python scripts/build_clusters.py`.")
            labels = None
        if labels is not None:
            cluster_frame = pd.DataFrame({"cluster": labels}).value_counts().reset_index(name="paper_count")
            st.subheader("Optional K-Means experiment")
            st.plotly_chart(style_chart(px.bar(cluster_frame, x="cluster", y="paper_count",
                                               title="MiniBatch K-Means cluster distribution")),
                            use_container_width=True)
    elif page.startswith("Saved Papers"):
        st.header("Saved Papers")
        st.markdown("<div class='rr-section-copy'>A private session reading list for papers you want to revisit.</div>", unsafe_allow_html=True)
        saved_rows = engine.papers[engine.papers["paper_id"].isin(st.session_state["saved_papers"])]
        if saved_rows.empty:
            st.info("No saved papers yet. Use the ☆ button on any search result to build your reading list.")
        else:
            st.write(f"{len(saved_rows)} saved paper(s)")
            for row_index, paper in saved_rows.iterrows():
                render_result(result_for_row(engine, row_index), engine)
    else:
        st.header("About ResearchRadar")
        st.markdown("""ResearchRadar demonstrates information retrieval with a reproducible academic corpus.

**TF-IDF** represents how important a term is to one document relative to the corpus. **Cosine similarity** compares the direction of sparse document vectors. **LDA** discovers probabilistic groups of co-occurring terms. Keywords are extracted from per-paper TF-IDF weights.

The source is the [ML-ArXiv-Papers dataset](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers). The documented fields are title and abstract, so this app intentionally does not invent authors, dates, citations, or paper links. Search is lexical and may miss semantically related wording.""")


if __name__ == "__main__":
    main()
