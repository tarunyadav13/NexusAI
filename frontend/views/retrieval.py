import streamlit as st

from components.citations import display_sources


def render():

    st.title("🔍 Retrieval Explorer")

    retrieval = st.session_state.get(
        "last_retrieval"
    )

    if not retrieval:

        st.info(
            "Ask a question in the Chat page to inspect retrieval results."
        )

        return

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dense Results",
        retrieval.get("dense_count", 0)
    )

    col2.metric(
        "Sparse Results",
        retrieval.get("sparse_count", 0)
    )

    col3.metric(
        "Fused Results",
        retrieval.get("fused_count", 0)
    )

    st.markdown("### Retrieved Documents")

    display_sources(
        retrieval.get("results", [])
    )