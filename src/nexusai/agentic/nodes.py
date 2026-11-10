import re
import logging

from nexusai.agentic.llm import GeminiLLM
from nexusai.agentic.state import AgenticRAGState
from nexusai.agentic.web_search import WebSearchService


logger = logging.getLogger(__name__)


class AgenticRAGNodes:

    def __init__(self, retriever):

        self.retriever = retriever

        self.llm = GeminiLLM()

        self.web_search = WebSearchService()

    # -----------------------------------------
    # QUERY ANALYSIS
    # -----------------------------------------

    def analyze_query(self, state):

        query = state["original_query"].strip()

        prompt = f"""
Classify this user query.

Question:
{query}

Return WEB if the question requires:
- Current information
- Recent developments
- News
- Live external information
- Information from websites
- Information not likely available in a static knowledge base

Otherwise return INTERNAL.

Return only one word: WEB or INTERNAL.
"""

        decision = self.llm.generate(prompt).strip().upper()

        use_web = decision.startswith("WEB")

        return {
            "current_query": query,
            "rewritten_query": query,
            "use_web": use_web
        }

    # -----------------------------------------
    # INTERNAL DOCUMENT RETRIEVAL
    # -----------------------------------------

    def retrieve_documents(self, state):

        query = state.get(
            "current_query",
            state["original_query"]
        )

        output = self.retriever.search(query)

        documents = output.get("results", [])

        normalized_documents = []

        for item in documents:

            normalized_documents.append({
                **item,
                "source_type": item.get(
                    "source_type",
                    "document"
                )
            })

        return {
            "documents": normalized_documents
        }

    # -----------------------------------------
    # EXTERNAL WEB SEARCH
    # -----------------------------------------

    def search_web(self, state):

        query = state.get(
            "current_query",
            state["original_query"]
        )

        try:

            result = self.web_search.search(
                query,
                max_results=5
            )

            web_results = result.get(
                "results",
                []
            )

            normalized_results = []

            for item in web_results:

                normalized_results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "source_type": "web"
                })

            return {
                "web_results": normalized_results
            }

        except Exception as exc:

            logger.exception(
                "External web search failed"
            )

            return {
                "web_results": [],
                "error": str(exc)
            }

    # -----------------------------------------
    # EVIDENCE GRADING
    # -----------------------------------------

    def grade_evidence(self, state):

        documents = state.get("documents", [])

        web_results = state.get("web_results", [])

        evidence = documents + web_results

        if not evidence:

            return {
                "relevance_score": 0.0,
                "evidence_sufficient": False
            }

        context = "\n\n".join(

            item.get("content", "")

            for item in evidence

            if item.get("content")

        )

        prompt = f"""
Evaluate whether the following evidence is sufficient
to answer the question accurately.

Question:
{state["original_query"]}

Evidence:
{context[:12000]}

Consider:
- Relevance
- Factual completeness
- Whether the evidence directly answers the question

Return ONLY a number between 0 and 1.
"""

        response = self.llm.generate(prompt)

        match = re.search(
            r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b",
            response
        )

        score = float(match.group()) if match else 0.0

        return {
            "relevance_score": score,
            "evidence_sufficient": score >= 0.50
        }

    # -----------------------------------------
    # QUERY REWRITING
    # -----------------------------------------

    def rewrite_query(self, state):

        prompt = f"""
Rewrite this question into a better retrieval query.

Original question:
{state["original_query"]}

Previous query:
{state.get("current_query", "")}

Return only the improved query.
"""

        rewritten = self.llm.generate(prompt).strip()

        if not rewritten:
            rewritten = state["original_query"]

        return {
            "current_query": rewritten,
            "retry_count": state.get("retry_count", 0) + 1
        }

    # -----------------------------------------
    # ANSWER GENERATION
    # -----------------------------------------

    def generate_answer(self, state):

        documents = state.get("documents", [])

        web_results = state.get("web_results", [])

        evidence = documents + web_results

        if not evidence:

            return {
                "answer": (
                    "I could not find relevant information "
                    "in the knowledge base or external web search."
                ),
                "citations": []
            }

        context_parts = []

        citations = []

        seen_sources = {}

        for item in evidence:

            source = (
                item.get("url")
                or item.get("source_url")
                or item.get("source")
                or "Unknown source"
            )

            source = str(source)

            title = item.get("title", "")

            content = item.get("content", "")

            if not content:
                continue

            # Deduplicate citations by source URL/path.
            if source in seen_sources:

                citation_id = seen_sources[source]

            else:

                citation_id = f"S{len(citations) + 1}"

                seen_sources[source] = citation_id

                citations.append({
                    "id": citation_id,
                    "source": source,
                    "title": title,
                    "source_type": item.get(
                        "source_type",
                        "document"
                    )
                })

            context_parts.append(
                f"[{citation_id}]\n"
                f"Title: {title}\n"
                f"Source: {source}\n"
                f"Source type: {item.get('source_type', 'document')}\n"
                f"Content: {content}"
            )

        if not context_parts:

            return {
                "answer": "No usable evidence was found.",
                "citations": []
            }

        context = "\n\n".join(context_parts)

        prompt = f"""
You are NexusAI, an evidence-grounded research assistant.

Answer the user's question using ONLY the supplied evidence.

IMPORTANT RULES:

1. Do not invent facts.
2. Cite factual claims using [S1], [S2], etc.
3. Use internal documents and external web sources when relevant.
4. Distinguish internal knowledge-base sources from live web sources.
5. If sources disagree, explain the disagreement.
6. If evidence is insufficient, explicitly acknowledge it.
7. Never invent citations.
8. Avoid repeating the same information.
9. Give a structured, clear answer.

Question:
{state["original_query"]}

Evidence:
{context}

Answer:
"""

        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "citations": citations
        }