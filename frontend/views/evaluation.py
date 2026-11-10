import streamlit as st
import json

from config import RESULTS_DIR


def load_results(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        st.error(f"Could not load {filename}: {exc}")
        return None


def get_summary(data):
    if not data:
        return {}

    return data.get("summary", data)


def get_metric(data, key, default=0):
    summary = get_summary(data)
    value = summary.get(key, default)

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_citation_coverage(answers):
    if not answers:
        return 0.0

    summary = get_summary(answers)

    if "citation_coverage" in summary:
        return float(summary["citation_coverage"])

    if "citation_reference_precision" in summary:
        return float(summary["citation_reference_precision"])

    individual = answers.get("individual_results", [])

    if not individual:
        return 0.0

    values = []

    for item in individual:
        if "citation_coverage" in item:
            values.append(float(item["citation_coverage"]))

        elif isinstance(item.get("citations"), dict):
            values.append(
                float(
                    item["citations"].get(
                        "citation_reference_precision", 0
                    )
                )
            )

    return sum(values) / len(values) if values else 0.0


def render():

    st.title("📊 NexusAI Evaluation Dashboard")

    st.caption(
        "Performance evaluation of NexusAI's retrieval, "
        "answer generation, and system latency."
    )

    retrieval = load_results("retrieval_results.json")
    answers = load_results("answer_results.json")
    benchmark = load_results("benchmark_results.json")

    # Refresh button
    if st.button("🔄 Refresh Evaluation Results"):
        st.rerun()

    # =========================================
    # RETRIEVAL PERFORMANCE
    # =========================================

    st.subheader("Retrieval Performance")

    if retrieval:

        cols = st.columns(4)

        metrics = [
            ("Precision@5", "precision_at_5"),
            ("Recall@5", "recall_at_5"),
            ("MRR", "mrr"),
            ("NDCG@5", "ndcg_at_5")
        ]

        for col, (label, key) in zip(cols, metrics):
            col.metric(
                label,
                f"{get_metric(retrieval, key):.4f}"
            )

        st.caption(
            f"Queries evaluated: {retrieval.get('total_queries', 0)}"
        )

    else:
        st.warning("Retrieval results not found.")

    st.divider()

    # =========================================
    # ANSWER QUALITY
    # =========================================

    st.subheader("Answer Quality")

    if answers:

        cols = st.columns(3)

        faithfulness = get_metric(answers, "faithfulness")
        relevance = get_metric(answers, "answer_relevance")
        citation_coverage = calculate_citation_coverage(answers)

        cols[0].metric("Faithfulness", f"{faithfulness:.4f}")
        cols[1].metric("Answer Relevance", f"{relevance:.4f}")
        cols[2].metric("Citation Coverage", f"{citation_coverage:.4f}")

        st.caption(
            f"Queries evaluated: {answers.get('total_queries', 0)}"
        )

    else:
        st.warning("Answer evaluation results not found.")

    st.divider()

    # =========================================
    # PERFORMANCE BENCHMARK
    # =========================================

    st.subheader("Performance Benchmark")

    if benchmark:

        cols = st.columns(4)

        cols[0].metric(
            "Total Queries",
            benchmark.get("summary", {}).get(
                "total_queries",
                benchmark.get("total_queries", 0)
            )
        )

        cols[1].metric(
            "Mean Latency",
            f"{get_metric(benchmark, 'mean_latency_seconds'):.4f} s"
        )

        cols[2].metric(
            "Median Latency",
            f"{get_metric(benchmark, 'median_latency_seconds'):.4f} s"
        )

        cols[3].metric(
            "Maximum Latency",
            f"{get_metric(benchmark, 'max_latency_seconds'):.4f} s"
        )

    else:
        st.warning("Benchmark results not found.")

    st.divider()

    # =========================================
    # EVALUATION DETAILS
    # =========================================

    st.subheader("Evaluation Details")

    with st.expander("Retrieval Evaluation Results"):
        st.json(retrieval or {})

    with st.expander("Answer Quality Results"):
        st.json(answers or {})

    with st.expander("Performance Benchmark Results"):
        st.json(benchmark or {})