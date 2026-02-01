"""Redis key patterns and stream names."""

# Streams (event bus)
STREAM_VISION_OBJECTS = "stream:vision:objects"
STREAM_VISION_UNKNOWN = "stream:vision:unknown"
STREAM_VISION_RESEARCHED = "stream:vision:researched"
STREAM_VISION_LABELS = "stream:vision:labels"

# Consumer groups
GROUP_DASHBOARD = "dashboard"
GROUP_RESEARCH = "research_worker"
GROUP_VOICE = "voice_worker"
GROUP_BACKEND = "backend"

# Key patterns
def label_key(label_id: str) -> str:
    return f"label:{label_id}"

def embedding_key(emb_id: str) -> str:
    return f"emb:{emb_id}"

def track_key(track_id: str) -> str:
    return f"track:{track_id}"

def objcard_key(unknown_id: str) -> str:
    return f"objcard:{unknown_id}"

# Vector index name
VECTOR_INDEX_NAME = "idx:embeddings"

# Metrics keys
METRICS_UNKNOWN_COUNT = "metrics:unknown_count"
METRICS_KNOWN_COUNT = "metrics:known_count"
METRICS_TOTAL_QUERIES = "metrics:total_queries"
