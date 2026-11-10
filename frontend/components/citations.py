import streamlit as st


def display_sources(results):

    st.markdown("### 📚 Retrieved Evidence")

    if not results:
        st.info("No supporting documents were retrieved.")
        return

    for index, item in enumerate(results, start=1):

        source = item.get("source", "Unknown source")

        content = item.get(
            "content",
            item.get("text", "")
        )

        score = item.get(
            "rerank_score",
            item.get("score", item.get("rrf_score", 0))
        )

        with st.expander(
            f"Source S{index} · {source}"
        ):

            st.write(content)

            st.caption(
                f"Relevance score: {score}"
            )