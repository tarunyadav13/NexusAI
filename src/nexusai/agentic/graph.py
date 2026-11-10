import os

from pathlib import Path
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

from nexusai.agentic.state import AgenticRAGState
from nexusai.agentic.nodes import AgenticRAGNodes
from nexusai.retrieval.hybrid_retriever import HybridRetriever


ROOT_DIR = Path(__file__).resolve().parents[3]

load_dotenv(ROOT_DIR / ".env")


def build_agentic_graph():

    max_retries = int(
        os.getenv("AGENTIC_MAX_RETRIES", "2")
    )

    retriever = HybridRetriever()

    nodes = AgenticRAGNodes(retriever)

    workflow = StateGraph(AgenticRAGState)

    workflow.add_node(
        "analyze_query",
        nodes.analyze_query
    )

    workflow.add_node(
        "retrieve",
        nodes.retrieve_documents
    )

    workflow.add_node(
        "web_search",
        nodes.search_web
    )

    workflow.add_node(
        "grade_evidence",
        nodes.grade_evidence
    )

    workflow.add_node(
        "rewrite_query",
        nodes.rewrite_query
    )

    workflow.add_node(
        "generate_answer",
        nodes.generate_answer
    )

    workflow.add_edge(
        START,
        "analyze_query"
    )

    workflow.add_edge(
        "analyze_query",
        "retrieve"
    )

    # Internal retrieval always happens first.
    # Web retrieval is conditional.

    def decide_web(state):

        if state.get("use_web", False):
            return "web_search"

        return "grade_evidence"

    workflow.add_conditional_edges(
        "retrieve",
        decide_web,
        {
            "web_search": "web_search",
            "grade_evidence": "grade_evidence"
        }
    )

    workflow.add_edge(
        "web_search",
        "grade_evidence"
    )

    def decide_next_step(state):

        if state.get("evidence_sufficient", False):
            return "generate_answer"

        if state.get("retry_count", 0) >= max_retries:
            return "generate_answer"

        return "rewrite_query"

    workflow.add_conditional_edges(
        "grade_evidence",
        decide_next_step,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query"
        }
    )

    workflow.add_edge(
        "rewrite_query",
        "retrieve"
    )

    workflow.add_edge(
        "generate_answer",
        END
    )

    return workflow.compile(), retriever