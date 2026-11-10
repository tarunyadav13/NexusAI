import os
import logging

from pathlib import Path
from dotenv import load_dotenv

from tavily import TavilyClient


ROOT_DIR = Path(__file__).resolve().parents[3]

load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger(__name__)


class WebSearchService:

    def __init__(self):

        api_key = os.getenv("TAVILY_API_KEY")

        self.client = (
            TavilyClient(api_key=api_key)
            if api_key
            else None
        )

    def search(self, query, max_results=5):

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if not self.client:
            raise ValueError(
                "TAVILY_API_KEY is missing. "
                "Configure it in your .env file."
            )

        max_results = max(
            1,
            min(int(max_results), 10)
        )

        try:

            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False
            )

            results = []

            for item in response.get("results", []):

                results.append({

                    "title": item.get("title", ""),

                    "url": item.get("url", ""),

                    "content": item.get("content", ""),

                    "score": item.get("score", 0.0),

                    "source_type": "web"

                })

            return {
                "query": query,
                "results": results,
                "count": len(results)
            }

        except Exception as exc:

            logger.exception("Web search failed")

            raise RuntimeError(
                f"Web search failed: {exc}"
            ) from exc