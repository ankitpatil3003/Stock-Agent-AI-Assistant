import logging
from typing import List, Dict, Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    QDRANT_AVAILABLE = True
except Exception:
    QDRANT_AVAILABLE = False


class VectorStore:
    """Abstract-ish interface used by KnowledgeAgent."""

    def add(self, ids: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]], documents: List[str]) -> None:
        raise NotImplementedError

    def query(self, query_vectors: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError


class MemoryVectorStore(VectorStore):
    """Simple in-memory store (for tiny datasets or tests)."""

    def __init__(self, dim: int):
        self.dim = dim
        self._vecs: Optional[np.ndarray] = None
        self._docs: List[str] = []
        self._meta: List[Dict[str, Any]] = []
        self._ids: List[str] = []

    def add(self, ids: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]], documents: List[str]) -> None:
        if self._vecs is None:
            self._vecs = embeddings.astype(np.float32)
        else:
            self._vecs = np.vstack([self._vecs, embeddings.astype(np.float32)])
        self._docs.extend(documents)
        self._meta.extend(metadatas)
        self._ids.extend(ids)

    def query(self, query_vectors: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        if self._vecs is None or len(self._docs) == 0:
            return []
        q = query_vectors.astype(np.float32)
        # cosine sim
        a = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        b = self._vecs / (np.linalg.norm(self._vecs, axis=1, keepdims=True) + 1e-9)
        sims = (a @ b.T).ravel()
        idx = np.argsort(-sims)[:k]
        out: List[Dict[str, Any]] = []
        for i in idx:
            out.append({
                "id": self._ids[i],
                "document": self._docs[i],
                "metadata": self._meta[i],
                "score": float(sims[i]),
            })
        return out


class QdrantVectorStore(VectorStore):
    """Qdrant backend."""

    def __init__(self, url: str, api_key: str, collection: str, dim: int, distance: str = "Cosine"):
        if not QDRANT_AVAILABLE:
            raise RuntimeError("qdrant-client not installed.")
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60.0)
        self.collection = collection
        self.dim = dim
        self._ensure_collection(distance)

    def _ensure_collection(self, distance: str) -> None:
        try:
            existing = self.client.get_collection(self.collection)
            # exists
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(size=self.dim, distance=getattr(qmodels.Distance, distance)),
            )

    def add(self, ids: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]], documents: List[str]) -> None:
        points = []
        for i, _id in enumerate(ids):
            points.append(
                qmodels.PointStruct(
                    id=_id,
                    vector=embeddings[i].astype(float).tolist(),
                    payload={
                        **(metadatas[i] or {}),
                        "document": documents[i],
                    },
                )
            )
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def query(self, query_vectors: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query_vectors[0].astype(float).tolist(),
            limit=max(1, k),
            with_payload=True,
        )
        out: List[Dict[str, Any]] = []
        for r in res:
            payload = r.payload or {}
            doc = payload.get("document", "")
            meta = {k: v for k, v in payload.items() if k != "document"}
            out.append({
                "id": r.id,
                "document": doc,
                "metadata": meta,
                "score": float(r.score),
            })
        return out


def make_store(backend: str, dim: int, url: str = "", api_key: str = "", collection: str = "", distance: str = "Cosine") -> VectorStore:
    backend = (backend or "").lower()
    if backend == "qdrant":
        return QdrantVectorStore(url=url, api_key=api_key, collection=collection, dim=dim, distance=distance)
    return MemoryVectorStore(dim=dim)
