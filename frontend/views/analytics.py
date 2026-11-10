import streamlit as st
import json

from config import RESULTS_DIR


def load_json(filename):

    path = RESULTS_DIR / filename

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return {}


def render():

    st.title("📊 NexusAI Analytics")

    st.caption("Monitor retrieval, answer quality and system performance.")

    retrieval = load_json("retrieval_results.json")
    answers = load_json("answer_results.json")
    benchmark = load_json("benchmark_results.json")

    st.subheader("Retrieval Performance")

    cols = st.columns(4)

    retrieval_metrics = [
        ("Precision@5", "precision_at_5"),
        ("Recall@5", "recall_at_5"),
        ("MRR", "mrr"),
        ("NDCG@5", "ndcg_at_5")
    ]

    for col, (label, key) in zip(cols, retrieval_metrics):

        value = retrieval.get(key)

        col.metric(
            label,
            f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"
        )

    st.divider()

    st.subheader("Answer Quality")

    cols = st.columns(3)

    answer_metrics = [
        ("Faithfulness", "faithfulness"),
        ("Answer Relevance", "answer_relevance"),
        ("Citation Coverage", "citation_coverage")
    ]

    for col, (label, key) in zip(cols, answer_metrics):

        value = answers.get(key)

        col.metric(
            label,
            f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"
        )

    st.divider()

    st.subheader("Performance Benchmark")

    cols = st.columns(4)

    benchmark_metrics = [
        ("Total Queries", "total_queries"),
        ("Mean Latency", "mean_latency_seconds"),
        ("Median Latency", "median_latency_seconds"),
        ("Maximum Latency", "max_latency_seconds")
    ]

    for col, (label, key) in zip(cols, benchmark_metrics):

        value = benchmark.get(key)

        col.metric(
            label,
            f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"
        )

    st.divider()

    st.subheader("Evaluation Data")

    with st.expander("Retrieval Results"):
        st.json(retrieval)

    with st.expander("Answer Quality Results"):
        st.json(answers)

    with st.expander("Benchmark Results"):
        st.json(benchmark)

    if st.button("Refresh Analytics"):
        st.rerun()