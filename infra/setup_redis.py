"""Create Redis vector index and consumer groups."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import redis
from redis.commands.search.field import VectorField, TagField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from shared.config import load_config
from shared.redis_keys import (
    VECTOR_INDEX_NAME,
    STREAM_VISION_OBJECTS,
    STREAM_VISION_UNKNOWN,
    STREAM_VISION_RESEARCHED,
    STREAM_VISION_LABELS,
    GROUP_DASHBOARD,
    GROUP_RESEARCH,
    GROUP_VOICE,
    GROUP_BACKEND,
)

EMBEDDING_DIM = 1280  # MobileNetV2 feature dim


def setup_vector_index(r: redis.Redis):
    """Create RediSearch vector index for embeddings."""
    try:
        r.ft(VECTOR_INDEX_NAME).dropindex(delete_documents=False)
        print(f"Dropped existing index: {VECTOR_INDEX_NAME}")
    except Exception:
        pass

    schema = (
        TagField("$.label_id", as_name="label_id"),
        TextField("$.label_name", as_name="label_name"),
        NumericField("$.created_at", as_name="created_at"),
        VectorField(
            "$.vector",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": EMBEDDING_DIM,
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="vector",
        ),
    )

    definition = IndexDefinition(prefix=["emb:"], index_type=IndexType.JSON)
    r.ft(VECTOR_INDEX_NAME).create_index(fields=schema, definition=definition)
    print(f"Created vector index: {VECTOR_INDEX_NAME} (dim={EMBEDDING_DIM})")


def setup_consumer_groups(r: redis.Redis):
    """Create streams and consumer groups."""
    streams_and_groups = [
        (STREAM_VISION_OBJECTS, [GROUP_DASHBOARD, GROUP_BACKEND]),
        (STREAM_VISION_UNKNOWN, [GROUP_DASHBOARD, GROUP_RESEARCH, GROUP_BACKEND]),
        (STREAM_VISION_RESEARCHED, [GROUP_DASHBOARD, GROUP_VOICE]),
        (STREAM_VISION_LABELS, [GROUP_DASHBOARD, GROUP_BACKEND]),
    ]

    for stream, groups in streams_and_groups:
        for group in groups:
            try:
                r.xgroup_create(stream, group, id="0", mkstream=True)
                print(f"Created group '{group}' on '{stream}'")
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    print(f"Group '{group}' already exists on '{stream}'")
                else:
                    raise


def main():
    config = load_config()
    r = redis.from_url(config.redis.url, decode_responses=True)

    print("Testing Redis connection...")
    r.ping()
    print("Redis connected!\n")

    setup_vector_index(r)
    print()
    setup_consumer_groups(r)
    print("\nRedis setup complete!")


if __name__ == "__main__":
    main()
