import streamlit as st
import subprocess
import sys

from config import ROOT_DIR, RAW_DIR


def render():

    st.title("📚 Knowledge Base")

    st.write(
        "Build your NexusAI knowledge base using PDFs and webpages."
    )

    st.divider()

    st.subheader("Upload documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    st.divider()

    st.subheader("Add webpage")

    url = st.text_input(
        "Enter webpage URL",
        placeholder="https://example.com/article"
    )

    if st.button(
        "Ingest URL",
        type="secondary"
    ):

        if not url.strip():

            st.warning("Enter a webpage URL.")
            return

        command = [
            sys.executable,
            "-c",
            (
                "from nexusai.ingestion.pipeline import IngestionPipeline;"
                f"result=IngestionPipeline().ingest_url({url!r});"
                "print(result.model_dump_json(indent=2));"
                "raise SystemExit(0 if result.status in "
                "('processed','duplicate') else 1)"
            )
        ]

        with st.spinner("Extracting and processing webpage..."):

            result = subprocess.run(
                command,
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=120
            )

        if result.returncode == 0:

            st.success("Webpage processed successfully.")

            st.code(result.stdout)

            st.info("Run indexing to make the webpage searchable.")

        else:

            st.error("Webpage ingestion failed.")

            st.code(result.stderr or result.stdout)

    st.divider()

    st.subheader("Upload documents")

    if st.button(
        "Process and Index Documents",
        type="primary"
    ):

        if not uploaded_files:

            st.warning("Upload at least one document.")
            return

        RAW_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        for uploaded_file in uploaded_files:

            destination = RAW_DIR / uploaded_file.name

            with open(destination, "wb") as file:
                file.write(uploaded_file.getbuffer())

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

                st.error(
                    f"Command failed: {command[1]}"
                )

                st.code(result.stderr)

                return

            if result.stdout:

                with st.expander("Execution logs"):

                    st.code(result.stdout)

        st.success(
            "Documents successfully processed and indexed!"
        )

    if st.button(
        "Index All Processed Sources",
        type="secondary"
    ):

        result = subprocess.run(
            [
                sys.executable,
                "scripts/index_documents.py"
            ],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:

            st.success("All sources indexed successfully.")

            st.code(result.stdout)

        else:

            st.error("Indexing failed.")

            st.code(result.stderr)