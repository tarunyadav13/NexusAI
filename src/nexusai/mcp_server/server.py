import logging
import json

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from nexusai.retrieval.hybrid_retriever import HybridRetriever
from nexusai.agentic.graph import build_agentic_graph
from nexusai.agentic.web_search import WebSearchService
from nexusai.ingestion.url_loader import URLLoader


logging.basicConfig(
    level=logging.INFO,
    stream=__import__("sys").stderr
)

logger = logging.getLogger(__name__)

mcp = FastMCP("NexusAI")

ROOT_DIR = Path(__file__).resolve().parents[3]

PROCESSED_DIR = ROOT_DIR / "data" / "processed"


# =========================================
# INTERNAL DOCUMENT SEARCH
# =========================================

@mcp.tool(description="Search the locally indexed knowledge base using hybrid retrieval. Returns relevant document chunks with source information.")
def search_documents(query: str, limit: int = 5) -> dict:

    if not query.strip():
        return {"error": "Query cannot be empty"}

    retriever = HybridRetriever(
        final_top_k=max(1, min(limit, 20))
    )

    try:

        results = retriever.search(query)

        return {
            "query": query,
            "results": results.get("results", []),
            "result_count": len(results.get("results", []))
        }

    except Exception as exc:

        logger.exception("Document search failed")

        return {"error": str(exc)}

    finally:

        retriever.close()


# =========================================
# AGENTIC NEXUSAI
# =========================================

@mcp.tool(description="Ask NexusAI a question using Agentic RAG. Retrieves internal documents, optionally searches the web, evaluates evidence, and returns a grounded answer with citations.")
def ask_nexusai(query: str, use_web: bool = True) -> dict:

    """
    Ask NexusAI using internal RAG and optionally
    external web search.
    """

    if not query.strip():
        return {"error": "Query cannot be empty"}

    graph, retriever = build_agentic_graph()

    try:

        result = graph.invoke({

            "original_query": query,

            "current_query": query,

            "rewritten_query": query,

            "documents": [],

            "web_results": [],

            "use_web": use_web,

            "relevance_score": 0.0,

            "evidence_sufficient": False,

            "retry_count": 0,

            "max_retries": 2,

            "answer": "",

            "citations": []

        })

        return {

            "query": query,

            "answer": result.get("answer", ""),

            "citations": result.get("citations", []),

            "evidence_score": result.get(
                "relevance_score",
                0.0
            ),

            "evidence_sufficient": result.get(
                "evidence_sufficient",
                False
            ),

            "retry_count": result.get(
                "retry_count",
                0
            ),

            "web_search_used": use_web,

            "web_results_count": len(
                result.get("web_results", [])
            )

        }

    except Exception as exc:

        logger.exception("Agentic RAG failed")

        return {"error": str(exc)}

    finally:

        retriever.close()


# =========================================
# LIST DOCUMENTS
# =========================================

@mcp.tool(description="List all documents indexed in the NexusAI knowledge base, including their titles, source paths, and source types.")
def list_documents() -> dict:

    documents = []

    if not PROCESSED_DIR.exists():

        return {
            "documents": [],
            "count": 0
        }

    for file_path in PROCESSED_DIR.glob("*.jsonl"):

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                for line in file:

                    if not line.strip():
                        continue

                    record = json.loads(line)

                    metadata = record.get(
                        "metadata",
                        {}
                    )

                    documents.append({

                        "document_id": record.get(
                            "document_id",
                            file_path.stem
                        ),

                        "source": metadata.get(
                            "source",
                            metadata.get("source_path", "")
                        ),

                        "source_type": metadata.get(
                            "source_type",
                            "document"
                        ),

                        "title": metadata.get(
                            "title",
                            ""
                        ),

                        "filename": file_path.name

                    })

        except Exception:

            logger.exception(
                "Could not read %s",
                file_path
            )

    return {
        "documents": documents,
        "count": len(documents)
    }


# =========================================
# DOCUMENT DETAILS
# =========================================

@mcp.tool(description="Retrieve metadata and source information for a specific indexed document using its document ID.")
def get_document_details(document_id: str) -> dict:

    if not document_id.strip():

        return {
            "error": "Document ID cannot be empty"
        }

    for file_path in PROCESSED_DIR.glob("*.jsonl"):

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                for line in file:

                    if not line.strip():
                        continue

                    record = json.loads(line)

                    current_id = record.get(
                        "document_id"
                    )

                    if str(current_id) == document_id:

                        return {

                            "document_id": current_id,

                            "metadata": record.get(
                                "metadata",
                                {}
                            ),

                            "filename": file_path.name

                        }

        except Exception:

            logger.exception(
                "Could not inspect %s",
                file_path
            )

    return {
        "error": f"Document '{document_id}' not found"
    }


# =========================================
# WEB MCP SEARCH
# =========================================

@mcp.tool(description="Search the live internet using Tavily and return relevant results with titles, URLs, summaries, and relevance scores.")
def search_web(query: str, max_results: int = 5) -> dict:

    """
    Search the live internet using Tavily.
    """

    if not query.strip():

        return {
            "error": "Query cannot be empty"
        }

    try:

        service = WebSearchService()

        return service.search(
            query=query,
            max_results=max(1, min(max_results, 10))
        )

    except Exception as exc:

        logger.exception(
            "Web MCP search failed"
        )

        return {
            "error": str(exc)
        }


# =========================================
# WEBPAGE READER
# =========================================

@mcp.tool(description="Fetch a webpage and extract its readable text, title, and URL for external research.")
def read_webpage(url: str) -> dict:

    """
    Read and extract webpage text from a URL.
    """

    try:

        loader = URLLoader()

        document = loader.load(url)

        page = document.pages[0]

        return {

            "url": url,

            "title": page.metadata.get(
                "title",
                document.filename
            ),

            "content": page.text,

            "source_type": "webpage"

        }

    except Exception as exc:

        logger.exception(
            "Webpage reading failed"
        )

        return {
            "error": str(exc)
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")