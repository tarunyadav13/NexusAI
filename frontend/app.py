import streamlit as st
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(FRONTEND_DIR))

from components.sidebar import render_sidebar

from views import (
    chat,
    knowledge_base,
    retrieval,
    agentic,
    evaluation,
    documents,
    research_assistant,
    analytics
)

st.set_page_config(
    page_title="NexusAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background-color: #0B1120;
    color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stMetric"] {
    background-color: #172033;
    padding: 18px;
    border-radius: 12px;
}

div.stButton > button {
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


page = render_sidebar()

if page == "Chat":
    chat.render()

elif page == "Documents":
    documents.render()

elif page == "Knowledge Base":
    knowledge_base.render()

elif page == "Research Assistant":
    research_assistant.render()

elif page == "Retrieval Explorer":
    retrieval.render()

elif page == "Agentic RAG":
    agentic.render()

elif page == "Analytics":
    analytics.render()

elif page == "Evaluation Dashboard":
    evaluation.render()