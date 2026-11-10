import logging
from pathlib import Path
import json

from nexusai.embeddings.generator import EmbeddingGenerator
from nexusai.vectorstore.qdrant_store import QdrantStore

from nexusai.retrieval.bm25_retriever import BM25Retriever
from nexusai.retrieval.fusion import reciprocal_rank_fusion
from nexusai.retrieval.reranker import BGEReranker


logger = logging.getLogger(__name__)


class HybridRetriever:

    def __init__(
        self,
        dense_top_k=10,
        sparse_top_k=10,
        fusion_top_k=10,
        final_top_k=5,
        use_reranker=True
    ):

        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.fusion_top_k = fusion_top_k
        self.final_top_k = final_top_k

        self.embedder = EmbeddingGenerator()
        self.vector_store = QdrantStore()

        self.use_reranker = use_reranker

        self.reranker = (
            BGEReranker()
            if use_reranker
            else None
        )

        self.bm25 = BM25Retriever(
            self._load_documents()
        )

    @staticmethod
    def _load_documents():

        root = Path(__file__).resolve().parents[3]

        processed_dir = root / "data" / "processed"

        documents = []

        for file_path in processed_dir.glob("*.jsonl"):

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                for line_number, line in enumerate(file):

                    if not line.strip():
                        continue

                    record = json.loads(line)

                    records = (
                        record.get("chunks", [])
                        if isinstance(record.get("chunks"), list)
                        else [record]
                    )

                    for item in records:

                        if not isinstance(item, dict):
                            continue

                        metadata = item.get("metadata", {})

                        if not isinstance(metadata, dict):
                            metadata = {}

                        content = next(
                            (
                                item[key]
                                for key in (
                                    "text",
                                    "content",
                                    "chunk_text",
                                    "page_content"
                                )
                                if isinstance(item.get(key), str)
                            ),
                            None
                        )

                        if not content:
                            continue

                        document_id = (
                            item.get("document_id")
                            or metadata.get("document_id")
                            or file_path.stem
                        )

                        chunk_index = (
                            item.get("chunk_index")
                            if item.get("chunk_index") is not None
                            else metadata.get(
                                "chunk_index",
                                line_number
                            )
                        )

                        chunk_id = (
                            f"{document_id}:{chunk_index}"
                        )

                        source = (
                            metadata.get("source")
                            or metadata.get("source_path")
                            or item.get("source")
                            or file_path.name
                        )

                        documents.append({

                            "chunk_id": str(chunk_id),

                            "content": content,

                            "source": str(source),

                            "document_id": str(document_id),

                            "chunk_index": str(chunk_index),

                            "source_type": metadata.get(
                                "source_type",
                                "document"
                            ),

                            "source_url": metadata.get(
                                "source_url"
                            ),

                            "title": metadata.get(
                                "title",
                                ""
                            ),

                            "page_numbers": metadata.get(
                                "page_numbers",
                                []
                            ),

                            "metadata": metadata

                        })

        return documents

    @staticmethod
    def _normalize_dense(results):

        normalized = []

        for result in results:

            metadata = result["metadata"]

            normalized.append({

                "chunk_id": (
                    f"{metadata.get('document_id', '')}:"
                    f"{metadata.get('chunk_index', '')}"
                ),

                "content": result["text"],

                "source": metadata.get(
                    "source",
                    metadata.get("source_path", "")
                ),

                "document_id": metadata.get(
                    "document_id",
                    ""
                ),

                "source_type": metadata.get(
                    "source_type",
                    "document"
                ),

                "source_url": metadata.get(
                    "source_url"
                ),

                "title": metadata.get(
                    "title",
                    ""
                ),

                "page_numbers": metadata.get(
                    "page_numbers",
                    []
                ),

                "metadata": metadata,

                "dense_score": result["score"]

            })

        return normalized

    def search(self, query):

        if not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(
            "Starting hybrid retrieval: %s",
            query
        )

        # 1. Dense retrieval

        query_vector = self.embedder.embed_query(query)

        dense_results = self.vector_store.search(
            query_vector,
            limit=self.dense_top_k
        )

        dense_results = self._normalize_dense(
            dense_results
        )

        # 2. Sparse retrieval

        sparse_results = self.bm25.search(
            query,
            top_k=self.sparse_top_k
        )

        # 3. Reciprocal Rank Fusion

        fused_results = reciprocal_rank_fusion(

            [dense_results, sparse_results],

            top_k=self.fusion_top_k
        )

        # 4. Cross-encoder reranking

        if self.reranker:

            final_results = self.reranker.rerank(

                query,

                fused_results,

                top_k=self.final_top_k
            )

        else:

            final_results = fused_results[:self.final_top_k]

        logger.info(
            "Hybrid retrieval completed: %s results",
            len(final_results)
        )

        return {

            "query": query,

            "dense_count": len(dense_results),

            "sparse_count": len(sparse_results),

            "fused_count": len(fused_results),

            "results": final_results

        }

    def close(self):

        self.embedder.close()

        self.vector_store.close()