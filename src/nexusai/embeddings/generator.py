import os
import time
import logging

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types


ROOT_DIR = Path(__file__).resolve().parents[3]

load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger(__name__)


class EmbeddingGenerator:

    def __init__(self):

        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing from .env"
            )

        self.model = os.getenv(
            "EMBEDDING_MODEL",
            "gemini-embedding-2"
        )

        self.dimension = int(
            os.getenv("EMBEDDING_DIMENSION", "768")
        )

        self.client = genai.Client(
            api_key=self.api_key
        )

    def _embed(self, contents):

        for attempt in range(3):

            try:

                response = self.client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimension
                    )
                )

                if not response.embeddings:
                    raise RuntimeError(
                        "Gemini returned empty embeddings"
                    )

                vectors = []

                for embedding in response.embeddings:

                    if embedding.values is None:
                        raise RuntimeError(
                            "Embedding values are missing"
                        )

                    vector = list(embedding.values)

                    if len(vector) != self.dimension:
                        raise ValueError(
                            f"Expected {self.dimension} dimensions, "
                            f"received {len(vector)}"
                        )

                    vectors.append(vector)

                return vectors

            except Exception as error:

                logger.warning(
                    "Embedding attempt %s failed: %s",
                    attempt + 1,
                    error
                )

                if attempt == 2:
                    raise

                time.sleep(2 ** attempt)

    def embed_documents(
        self,
        documents: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:

        if not documents:
            return []

        all_embeddings = []

        for start in range(0, len(documents), batch_size):

            batch = documents[start:start + batch_size]

            contents = [
                types.Content(
                    parts=[
                        types.Part.from_text(
                            text=f"title: none | text: {document}"
                        )
                    ]
                )
                for document in batch
            ]

            vectors = self._embed(contents)

            if len(vectors) != len(batch):
                raise RuntimeError(
                    "Embedding count does not match input count"
                )

            all_embeddings.extend(vectors)

            logger.info(
                "Embedded %s/%s documents",
                len(all_embeddings),
                len(documents)
            )

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:

        if not query.strip():
            raise ValueError("Query cannot be empty")

        contents = (
            f"task: search result | query: {query}"
        )

        vectors = self._embed(contents)

        return vectors[0]

    def close(self):

        self.client.close()