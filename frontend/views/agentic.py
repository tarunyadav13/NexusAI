import streamlit as st
import subprocess
import sys
import json

from config import ROOT_DIR
from views.chat import load_retriever


def release_qdrant_lock():

    try:

        retriever = load_retriever()

        retriever.close()

    finally:

        load_retriever.clear()


def render():

    st.title("🤖 Agentic RAG")

    st.write(
        "Ask NexusAI questions using internal documents "
        "and external web research."
    )

    query = st.text_area(
        "Enter your question",
        placeholder=(
            "Compare the concepts in my documents "
            "with recent developments on the web."
        )
    )

    if st.button(
        "Run Agentic RAG",
        type="primary"
    ):

        if not query.strip():

            st.warning("Enter a question first.")

            return

        command = [
            sys.executable,
            "scripts/run_agentic_rag.py",
            "--query",
            query
        ]

        try:

            with st.spinner("Preparing Qdrant..."):

                release_qdrant_lock()

            with st.spinner(
                "NexusAI is researching your question..."
            ):

                result = subprocess.run(
                    command,
                    cwd=ROOT_DIR,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

            if result.returncode != 0:

                st.error("Agentic RAG failed.")

                st.code(result.stderr)

                return

            st.success("Research completed.")

            try:

                output = json.loads(result.stdout)

                st.markdown("### Answer")

                st.markdown(
                    output.get("answer", "")
                )

                citations = output.get(
                    "citations",
                    []
                )

                if citations:

                    st.markdown("### Sources")

                    for citation in citations:

                        source = citation.get(
                            "source",
                            ""
                        )

                        title = citation.get(
                            "title",
                            source
                        )

                        if source.startswith("http"):

                            st.markdown(
                                f"- [{title}]({source})"
                            )

                        else:

                            st.write(
                                f"- {title}: {source}"
                            )

                with st.expander("Execution details"):

                    st.json({
                        "evidence_score": output.get(
                            "evidence_score"
                        ),
                        "evidence_sufficient": output.get(
                            "evidence_sufficient"
                        ),
                        "retry_count": output.get(
                            "retry_count"
                        )
                    })

            except json.JSONDecodeError:

                st.markdown("### Execution Output")

                st.code(result.stdout)

        except subprocess.TimeoutExpired:

            st.error("Execution timed out after 5 minutes.")

        except Exception as error:

            st.error(
                f"Unable to execute Agentic RAG: {error}"
            )