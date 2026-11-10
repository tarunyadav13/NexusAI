import streamlit as st
from streamlit_option_menu import option_menu


def render_sidebar():

    with st.sidebar:

        st.markdown(
            """
            <h1 style="text-align:center;">
                🧠 NexusAI
            </h1>
            <p style="text-align:center;color:gray;">
                Agentic RAG Platform
            </p>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        selected = option_menu(
            menu_title="Workspace",

            options=[
                "Chat",
                "Documents",
                "Knowledge Base",
                "Research Assistant",
                "Retrieval Explorer",
                "Agentic RAG",
                "Analytics",
                "Evaluation Dashboard"
            ],

            icons=[
                "chat-dots",
                "file-earmark-text",
                "database",
                "robot",
                "search",
                "diagram-3",
                "bar-chart",
                "clipboard-data"
            ],

            default_index=0,

            styles={
                "container": {
                    "padding": "5px",
                    "background-color": "#111827"
                },

                "icon": {
                    "color": "#60A5FA",
                    "font-size": "18px"
                },

                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "5px"
                },

                "nav-link-selected": {
                    "background-color": "#2563EB"
                }
            }
        )

        st.divider()

        st.markdown(
            """
            <div style="text-align:center;color:gray;">
                NexusAI v1.0<br><br>
                Gemini · Qdrant · LangGraph · MCP
            </div>
            """,
            unsafe_allow_html=True
        )

        return selected