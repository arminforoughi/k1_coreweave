"""Vector memory with Redis KNN for object recognition."""
import json
import time
import uuid
import numpy as np
import redis
from redis.commands.search.query import Query
from typing import Optional
from shared.redis_keys import VECTOR_INDEX_NAME, label_key, embedding_key


class VectorMemory:
    """Manages object embeddings in Redis with KNN lookup."""

    def __init__(self, redis_client: redis.Redis, embedding_dim: int = 1280):
        self.r = redis_client
        self.dim = embedding_dim

    def store_embedding(
        self,
        vector: np.ndarray,
        label_id: str,
        label_name: str,
        instance_id: Optional[str] = None,
    ) -> str:
        """Store an embedding vector in Redis."""
        emb_id = str(uuid.uuid4())
        key = embedding_key(emb_id)

        doc = {
            "vector": vector.astype(np.float32).tolist(),
            "label_id": label_id,
            "label_name": label_name,
            "instance_id": instance_id or "",
            "created_at": time.time(),
        }

        self.r.json().set(key, "$", doc)
        return emb_id

    def knn_lookup(self, vector: np.ndarray, k: int = 5) -> list[dict]:
        """Find k nearest embeddings. Returns list of {label_id, label_name, score, emb_id}."""
        query_vector = vector.astype(np.float32).tobytes()

        q = (
            Query(f"*=>[KNN {k} @vector $query_vec AS score]")
            .sort_by("score")
            .return_fields("label_id", "label_name", "score")
            .dialect(2)
        )

        try:
            results = self.r.ft(VECTOR_INDEX_NAME).search(
                q, query_params={"query_vec": query_vector}
            )
        except Exception:
            return []

        matches = []
        for doc in results.docs:
            # Redis cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - (distance/2)
            distance = float(doc.score)
            similarity = 1.0 - (distance / 2.0)
            matches.append({
                "emb_id": doc.id.replace("emb:", ""),
                "label_id": doc.label_id,
                "label_name": doc.label_name,
                "similarity": similarity,
            })

        return matches

    def learn_label(
        self,
        label_name: str,
        vectors: list[np.ndarray],
        label_id: Optional[str] = None,
    ) -> str:
        """Store multiple exemplar embeddings under a label."""
        if label_id is None:
            label_id = str(uuid.uuid4())

        # Store label metadata
        lkey = label_key(label_id)
        self.r.hset(lkey, mapping={
            "name": label_name,
            "created_at": str(time.time()),
            "n_examples": str(len(vectors)),
        })

        # Store each embedding
        for vec in vectors:
            self.store_embedding(vec, label_id, label_name)

        return label_id

    def get_all_labels(self) -> list[dict]:
        """Get all learned labels."""
        labels = []
        cursor = 0
        while True:
            cursor, keys = self.r.scan(cursor, match="label:*", count=100)
            for key in keys:
                data = self.r.hgetall(key)
                if data:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    label_id = key_str.replace("label:", "")
                    # Decode bytes values from binary redis client
                    def _dec(v):
                        return v.decode() if isinstance(v, bytes) else v
                    labels.append({
                        "label_id": label_id,
                        "name": _dec(data.get(b"name", data.get("name", ""))),
                        "n_examples": int(_dec(data.get(b"n_examples", data.get("n_examples", 0)))),
                        "created_at": float(_dec(data.get(b"created_at", data.get("created_at", 0)))),
                    })
            if cursor == 0:
                break
        return labels

    def get_memory_count(self) -> int:
        """Count total embeddings in memory."""
        try:
            result = self.r.execute_command("FT.INFO", VECTOR_INDEX_NAME)
            # Result is a flat list: [key, value, key, value, ...]
            if isinstance(result, list):
                for i in range(0, len(result) - 1, 2):
                    k = result[i]
                    if isinstance(k, bytes):
                        k = k.decode()
                    if k == "num_docs":
                        v = result[i + 1]
                        return int(v.decode() if isinstance(v, bytes) else v)
            return 0
        except Exception:
            return 0


class GatingLogic:
    """Decides known/uncertain/unknown based on KNN results."""

    def __init__(
        self,
        known_threshold: float = 0.7,
        unknown_threshold: float = 0.4,
        margin_threshold: float = 0.15,
    ):
        self.known_threshold = known_threshold
        self.unknown_threshold = unknown_threshold
        self.margin_threshold = margin_threshold

    def decide(self, matches: list[dict]) -> tuple[str, Optional[str], float]:
        """Decide state based on KNN matches.

        Returns: (state, label_name, top_similarity)
            state: "known", "uncertain", or "unknown"
        """
        if not matches:
            return "unknown", None, 0.0

        s1 = matches[0]["similarity"]
        s2 = matches[1]["similarity"] if len(matches) > 1 else 0.0
        margin = s1 - s2

        top_label = matches[0]["label_name"]

        if s1 >= self.known_threshold and margin >= self.margin_threshold:
            return "known", top_label, s1
        elif s1 >= self.unknown_threshold:
            # Uncertain: return None so backend uses YOLO class, not weak KNN match
            return "uncertain", None, s1
        else:
            return "unknown", None, s1
