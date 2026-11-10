import streamlit as st

from nexusai.retrieval.hybrid_retriever import HybridRetriever
from nexusai.agentic.llm import GeminiLLM

from components.citations import display_sources


@st.cache_resource
def load_retriever():

    return HybridRetriever(
        dense_top_k=10,
        sparse_top_k=10,
        fusion_top_k=10,
        final_top_k=5,
        use_reranker=True
    )


@st.cache_resource
def load_llm():

    return GeminiLLM()


def generate_answer(query, results):

    context = ""

    for index, item in enumerate(results, start=1):

        content = item.get(
            "content",
            item.get("text", "")
        )

        context += (
            f"\n[S{index}]\n"
            f"Source: {item.get('source', 'Unknown')}\n"
            f"Content: {content}\n"
        )

    prompt = f"""
You are NexusAI, an intelligent document question-answering assistant.

Answer the user's question using ONLY the provided evidence.

Rules:
1. Do not invent facts.
2. Cite claims using [S1], [S2], etc.
3. If evidence is insufficient, clearly state that.
4. Give a clear and structured answer.

Question:
{query}

Evidence:
{context}

Answer:
"""

    return load_llm().generate(prompt)


def render():

    st.title("💬 NexusAI Chat")

    st.caption(
        "Ask questions about your indexed documents."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant":

                if message.get("sources"):
                    display_sources(message["sources"])

    query = st.chat_input(
        "Ask NexusAI anything about your documents..."
    )

    if query:

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        try:

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching documents and generating answer..."
                ):

                    retriever = load_retriever()

                    retrieval = retriever.search(query)

                    results = retrieval.get(
                        "results",
                        []
                    )

                    answer = generate_answer(
                        query,
                        results
                    )

                    st.markdown(answer)

                    display_sources(results)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": results
                    })

                    st.session_state.last_retrieval = retrieval

        except Exception as error:

            st.error(
                f"Unable to generate response: {error}"
            )