import streamlit as st
import subprocess
import sys

from config import ROOT_DIR


def render():

    st.title("🔬 Research Assistant")

    st.caption(
        "Ask complex questions and let NexusAI retrieve and synthesize "
        "information from your indexed documents."
    )

    query = st.text_area(
        "Research question",
        placeholder="Explain the important concepts and relationships "
                    "discussed in my documents...",
        height=130
    )

    if st.button("Start Research", type="primary"):

        if not query.strip():
            st.warning("Enter a research question.")
            return

        command = [
            sys.executable,
            "scripts/run_agentic_rag.py",
            "--query",
            query
        ]

        with st.spinner("Research agent is working..."):

            try:

                result = subprocess.run(
                    command,
                    cwd=ROOT_DIR,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode == 0:

                    st.success("Research completed!")

                    st.subheader("Research Output")
                    st.code(result.stdout)

                else:

                    st.error("Research execution failed.")
                    st.code(result.stderr)

            except subprocess.TimeoutExpired:

                st.error("Research timed out after 5 minutes.")