"""FastAPI backend for the self-improving vision agent."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import uuid
import numpy as np
import redis
import weave
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from shared.config import load_config
from shared.redis_keys import (
    STREAM_VISION_OBJECTS,
    STREAM_VISION_UNKNOWN,
    STREAM_VISION_LABELS,
    METRICS_UNKNOWN_COUNT,
    METRICS_KNOWN_COUNT,
    METRICS_TOTAL_QUERIES,
    label_key,
    embedding_key,
    VECTOR_INDEX_NAME,
)
from shared.events import ObjectEvent, UnknownEvent, LabelEvent
from perception.memory import VectorMemory

load_dotenv()
config = load_config()

# Init Weave
weave.init(config.weave.project)

# Init Redis
r = redis.from_url(config.redis.url, decode_responses=True)
r_binary = redis.from_url(config.redis.url, decode_responses=False)

# Init memory
memory = VectorMemory(r_binary, embedding_dim=1280)

app = FastAPI(title="OpenClawdIRL API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---

class LabelRequest(BaseModel):
    track_id: Optional[str] = None
    unknown_event_id: Optional[str] = None
    label_name: str


class RenameRequest(BaseModel):
    label_id: str
    new_name: str


# --- Endpoints ---

@app.get("/health")
def health():
    try:
        r.ping()
        return {"status": "ok", "redis": "connected"}
    except Exception as e:
        return {"status": "error", "redis": str(e)}


@app.get("/events/objects")
def get_object_events(count: int = 50):
    """Get recent object events from the stream."""
    try:
        entries = r.xrevrange(STREAM_VISION_OBJECTS, count=count)
        events = []
        for entry_id, data in entries:
            data["stream_id"] = entry_id
            events.append(data)
        return {"events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/events/unknown")
def get_unknown_events(count: int = 50):
    """Get recent unknown events from the stream."""
    try:
        entries = r.xrevrange(STREAM_VISION_UNKNOWN, count=count)
        events = []
        for entry_id, data in entries:
            data["stream_id"] = entry_id
            events.append(data)
        return {"events": events}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@weave.op()
@app.post("/label")
def label_object(req: LabelRequest):
    """Teach the system a new label for an object.

    Finds the object's embedding from recent unknown events and stores it in memory.
    """
    # Find the embedding from unknown events
    embedding = None
    thumbnail = ""

    entries = r.xrevrange(STREAM_VISION_UNKNOWN, count=200)
    for entry_id, data in entries:
        if req.track_id and data.get("track_id") == req.track_id:
            emb_str = data.get("embedding", "")
            if emb_str:
                embedding = np.array(json.loads(emb_str), dtype=np.float32)
            thumbnail = data.get("thumbnail_b64", "")
            break
        if req.unknown_event_id and data.get("event_id") == req.unknown_event_id:
            emb_str = data.get("embedding", "")
            if emb_str:
                embedding = np.array(json.loads(emb_str), dtype=np.float32)
            thumbnail = data.get("thumbnail_b64", "")
            break

    if embedding is None:
        # Also check object events
        obj_entries = r.xrevrange(STREAM_VISION_OBJECTS, count=200)
        for entry_id, data in obj_entries:
            if req.track_id and data.get("track_id") == req.track_id:
                emb_str = data.get("embedding", "")
                if emb_str:
                    embedding = np.array(json.loads(emb_str), dtype=np.float32)
                thumbnail = data.get("thumbnail_b64", "")
                break

    if embedding is None:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding found for track_id={req.track_id} or event_id={req.unknown_event_id}",
        )

    # Store in memory
    label_id = memory.learn_label(req.label_name, [embedding])

    # Emit label event
    label_event = LabelEvent(
        track_id=req.track_id or "",
        label_name=req.label_name,
        label_id=label_id,
    )
    r.xadd(STREAM_VISION_LABELS, label_event.to_redis(), maxlen=500)

    return {
        "label_id": label_id,
        "label_name": req.label_name,
        "message": f"Learned '{req.label_name}' with 1 exemplar",
    }


@app.get("/memory")
def get_memory():
    """Get all learned labels and their stats."""
    labels = memory.get_all_labels()
    total_embeddings = memory.get_memory_count()
    return {
        "labels": labels,
        "total_embeddings": total_embeddings,
    }


@app.post("/rename")
def rename_label(req: RenameRequest):
    """Rename an existing label."""
    lkey = label_key(req.label_id)
    if not r.exists(lkey):
        raise HTTPException(status_code=404, detail=f"Label {req.label_id} not found")

    r.hset(lkey, "name", req.new_name)

    # Update label_name in all associated embeddings
    cursor = 0
    while True:
        cursor, keys = r_binary.scan(cursor, match=b"emb:*", count=100)
        for key in keys:
            doc = r_binary.json().get(key.decode())
            if doc and doc.get("label_id") == req.label_id:
                r_binary.json().set(key.decode(), "$.label_name", req.new_name)
        if cursor == 0:
            break

    return {"label_id": req.label_id, "new_name": req.new_name}


@app.get("/metrics")
def get_metrics():
    """Get improvement metrics."""
    unknown_count = int(r.get(METRICS_UNKNOWN_COUNT) or 0)
    known_count = int(r.get(METRICS_KNOWN_COUNT) or 0)
    total_queries = int(r.get(METRICS_TOTAL_QUERIES) or 0)

    # Calculate rates
    recognition_rate = known_count / total_queries if total_queries > 0 else 0.0
    unknown_rate = unknown_count / total_queries if total_queries > 0 else 0.0

    # Get time-series from metrics stream (if we add one later)
    memory_count = memory.get_memory_count()
    label_count = len(memory.get_all_labels())

    return {
        "unknown_count": unknown_count,
        "known_count": known_count,
        "total_queries": total_queries,
        "recognition_rate": round(recognition_rate, 4),
        "unknown_rate": round(unknown_rate, 4),
        "memory_size": memory_count,
        "label_count": label_count,
    }


@app.get("/metrics/history")
def get_metrics_history():
    """Get metrics over time (sampled from label events)."""
    # Return label events as a proxy for improvement over time
    entries = r.xrange(STREAM_VISION_LABELS, count=100)
    history = []
    for entry_id, data in entries:
        history.append({
            "timestamp": float(data.get("timestamp", 0)),
            "label_name": data.get("label_name", ""),
            "event": "label_added",
        })
    return {"history": history}


def start_server():
    import uvicorn
    uvicorn.run(
        "backend.api:app",
        host=config.backend.host,
        port=config.backend.port,
        reload=False,
    )


if __name__ == "__main__":
    start_server()
