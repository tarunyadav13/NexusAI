import streamlit as st
import subprocess
import sys

from config import ROOT_DIR, RAW_DIR


def render():

    st.title("📄 Document Management")

    st.caption(
        "Upload documents and add them to your NexusAI knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if st.button("Upload and Index", type="primary"):

        if not uploaded_files:
            st.warning("Please upload at least one document.")
            return

        RAW_DIR.mkdir(parents=True, exist_ok=True)

        for uploaded_file in uploaded_files:

            destination = RAW_DIR / uploaded_file.name

            with open(destination, "wb") as file:
                file.write(uploaded_file.getbuffer())

        st.success(f"{len(uploaded_files)} document(s) uploaded.")

        commands = [
            [
                sys.executable,
                "scripts/ingest_documents.py",
                "--input",
                "data/raw"
            ],
            [
                sys.executable,
                "scripts/index_documents.py"
            ]
        ]

        for command in commands:

            with st.spinner(f"Running {command[1]}..."):

                result = subprocess.run(
                    command,
                    cwd=ROOT_DIR,
                    capture_output=True,
                    text=True
                )

            if result.returncode != 0:
                st.error("Processing failed.")
                st.code(result.stderr)
                return

            with st.expander(f"{command[1]} logs"):
                st.code(result.stdout)

        st.success("Documents successfully indexed!")