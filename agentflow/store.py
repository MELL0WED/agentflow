from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from agentflow.chunking import Chunk
from agentflow.config import Settings


class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
        )
        return vectors.tolist()


class VectorStore:
    def __init__(self, settings: Settings, dim: int):
        self.is_server = bool(settings.qdrant_url)
        if self.is_server:
            self.client = QdrantClient(url=settings.qdrant_url)
        else:
            self.client = QdrantClient(path=settings.qdrant_path)
        self.collection = settings.collection
        self.dim = dim

    def recreate(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )
        if self.is_server:
            self.client.create_payload_index(
                self.collection, field_name="doc_id", field_schema=PayloadSchemaType.KEYWORD
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], batch_size: int = 256) -> None:
        for i in range(0, len(chunks), batch_size):
            points = [
                PointStruct(id=i + j, vector=vec, payload=chunk.payload())
                for j, (chunk, vec) in enumerate(zip(chunks[i : i + batch_size], vectors[i : i + batch_size]))
            ]
            self.client.upsert(self.collection, points=points)

    @staticmethod
    def _doc_filter(doc_id: str) -> Filter:
        return Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])

    def search(self, doc_id: str, vector: list[float], k: int) -> list[dict]:
        result = self.client.query_points(
            self.collection,
            query=vector,
            query_filter=self._doc_filter(doc_id),
            limit=k,
            with_payload=True,
        )
        return [{**p.payload, "score": p.score} for p in result.points]

    def has_document(self, doc_id: str) -> bool:
        return self.client.count(self.collection, count_filter=self._doc_filter(doc_id), exact=True).count > 0

    def close(self) -> None:
        self.client.close()