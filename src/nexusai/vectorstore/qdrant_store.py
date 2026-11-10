import os
import logging

from pathlib import Path
from typing import List, Dict
from uuid import uuid5, NAMESPACE_DNS

from dotenv import load_dotenv

from qdrant_client import QdrantClient, models


ROOT_DIR = Path(__file__).resolve().parents[3]

load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger(__name__)


class QdrantStore:

    def __init__(self):

        self.collection_name = os.getenv(
            "QDRANT_COLLECTION",
            "nexusai_documents"
        )

        self.dimension = int(
            os.getenv("EMBEDDING_DIMENSION", "768")
        )

        qdrant_url = os.getenv("QDRANT_URL")

        if qdrant_url:

            self.client = QdrantClient(
                url=qdrant_url,
                api_key=os.getenv("QDRANT_API_KEY")
            )

        else:

            storage_path = ROOT_DIR / os.getenv(
                "QDRANT_PATH",
                "data/vectorstore"
            )

            storage_path.mkdir(
                parents=True,
                exist_ok=True
            )

            self.client = QdrantClient(
                path=str(storage_path)
            )

        self._initialize_collection()

    def _initialize_collection(self):

        if not self.client.collection_exists(
            self.collection_name
        ):

            self.client.create_collection(

                collection_name=self.collection_name,

                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE
                )
            )

            logger.info(
                "Created Qdrant collection: %s",
                self.collection_name
            )

        else:

            collection = self.client.get_collection(
                self.collection_name
            )

            vector_config = collection.config.params.vectors

            if isinstance(vector_config, dict):
                raise ValueError(
                    "NexusAI expects a single unnamed vector collection"
                )

            existing_dimension = vector_config.size

            if existing_dimension != self.dimension:

                raise ValueError(
                    f"Vector dimension mismatch. "
                    f"Collection={existing_dimension}, "
                    f"Application={self.dimension}. "
                    f"Use a new collection when changing models."
                )

            logger.info(
                "Connected to existing Qdrant collection"
            )

    @staticmethod
    def _point_id(metadata):

        identity = (
            f"{metadata.get('document_id', '')}:"
            f"{metadata.get('chunk_index', '')}:"
            f"{metadata.get('text', '')}"
        )

        return str(
            uuid5(NAMESPACE_DNS, identity)
        )

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict]
    ):

        if not (
            len(texts) == len(embeddings) == len(metadata)
        ):
            raise ValueError(
                "Texts, embeddings and metadata must have equal lengths"
            )

        if not texts:
            return 0

        points = []

        for text, vector, meta in zip(
            texts,
            embeddings,
            metadata
        ):

            if len(vector) != self.dimension:
                raise ValueError(
                    "Invalid embedding dimension"
                )

            payload = {
                **meta,
                "text": text
            }

            points.append(

                models.PointStruct(

                    id=self._point_id(payload),

                    vector=vector,

                    payload=payload
                )
            )

        self.client.upsert(

            collection_name=self.collection_name,

            points=points,

            wait=True
        )

        logger.info(
            "Stored %s vectors in Qdrant",
            len(points)
        )

        return len(points)

    def search(
        self,
        query_vector: List[float],
        limit: int = 10
    ):

        if len(query_vector) != self.dimension:
            raise ValueError(
                "Query embedding dimension mismatch"
            )

        results = self.client.query_points(

            collection_name=self.collection_name,

            query=query_vector,

            limit=limit,

            with_payload=True
        )

        formatted_results = []

        for point in results.points:

            formatted_results.append({

                "id": str(point.id),

                "score": float(point.score),

                "text": point.payload.get("text", ""),

                "metadata": point.payload

            })

        return formatted_results

    def close(self):

        self.client.close()