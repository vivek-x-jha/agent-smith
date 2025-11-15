"""ChromaDB-backed semantic memory utilities."""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import Iterable, Sequence
from uuid import uuid4

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models import Collection
from chromadb.utils.embedding_functions import EmbeddingFunction

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


class LightweightEmbeddingFunction(EmbeddingFunction):
    """Deterministic hashing-based embeddings for local execution."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def __call__(self, input: Sequence[str]) -> list[list[float]]:  # type: ignore[override]
        vectors: list[list[float]] = []
        for text in input:
            accumulator = [0.0] * self.dimensions
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                for idx in range(self.dimensions):
                    accumulator[idx] += digest[idx] / 255.0
            norm = math.sqrt(sum(value * value for value in accumulator)) or 1.0
            vectors.append([value / norm for value in accumulator])
        return vectors


@lru_cache(maxsize=1)
def get_client() -> ClientAPI:
    """Return a cached persistent Chroma client."""

    settings = get_settings()
    return chromadb.PersistentClient(path=str(settings.chroma_path))


@lru_cache(maxsize=1)
def get_collection(name: str = "agent-smith-resources") -> Collection:
    """Return (and lazily create) the shared resource collection."""

    client = get_client()
    embedding_function = LightweightEmbeddingFunction()
    return client.get_or_create_collection(name=name, embedding_function=embedding_function)


def _resource_payload(resource: "Resource") -> tuple[str, str, dict[str, str | int | float | None]]:
    """Build payload tuple for Chroma upsert."""

    doc = resource.content or resource.snippet or resource.title
    metadata = {
        "goal_id": resource.goal_id,
        "plan_item_id": resource.plan_item_id,
        "title": resource.title,
        "url": resource.url,
        "source": resource.source,
    }
    vector_id = resource.vector_id or f"resource-{resource.id or uuid4()}"
    return vector_id, doc, metadata


def upsert_resources(resources: Iterable["Resource"]) -> list[str]:
    """Persist resources into the vector store and return their vector ids."""

    collection = get_collection()
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str | int | float | None]] = []

    from ..models import Resource  # Local import to avoid circular dependency

    for resource in resources:
        if not isinstance(resource, Resource):  # Safety check for callers
            raise TypeError("upsert_resources expects Resource instances")
        vector_id, document, metadata = _resource_payload(resource)
        resource.vector_id = vector_id
        ids.append(vector_id)
        documents.append(document)
        metadatas.append(metadata)

    if not ids:
        return []

    logger.info("vector_upsert", count=len(ids))
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return ids


def search_resources(query: str, goal_id: int | None = None, limit: int = 5) -> list[dict[str, object]]:
    """Search the vector store for semantically similar resources."""

    collection = get_collection()
    where = {"goal_id": goal_id} if goal_id is not None else None
    logger.info("vector_search", query=query, goal_id=goal_id, limit=limit)
    result = collection.query(query_texts=[query], n_results=limit, where=where)

    matches: list[dict[str, object]] = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for idx, vector_id in enumerate(ids):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        matches.append(
            {
                "vector_id": vector_id,
                "document": documents[idx] if idx < len(documents) else None,
                "metadata": metadata,
                "distance": distances[idx] if idx < len(distances) else None,
            }
        )

    return matches


__all__ = ["get_collection", "search_resources", "upsert_resources"]
