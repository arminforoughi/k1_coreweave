"""FastAPI backend — runs on laptop. Receives crops from Jetson, does all the thinking."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
import base64
import threading
import subprocess
import numpy as np
import redis
import weave
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image

from shared.config import load_config
from shared.redis_keys import (
    STREAM_VISION_OBJECTS,
    STREAM_VISION_UNKNOWN,
    STREAM_VISION_LABELS,
    STREAM_VISION_RESEARCHED,
    METRICS_UNKNOWN_COUNT,
    METRICS_KNOWN_COUNT,
    METRICS_TOTAL_QUERIES,
    label_key,
)
from shared.events import ObjectEvent, UnknownEvent, LabelEvent
from perception.embedder import Embedder
from perception.memory import VectorMemory, GatingLogic
from perception.tracker import SimpleTracker

load_dotenv()
config = load_config()

# Init Weave for observability tracing
entity_name = config.weave.entity
if entity_name and config.weave.api_key:
    try:
        weave.init(f"{entity_name}/{config.weave.project}")
        print(f"Weave tracing enabled: {entity_name}/{config.weave.project}")
    except Exception as e:
        print(f"Weave init failed ({e})")
else:
    print("Weave disabled — set WANDB_API_KEY + WANDB_ENTITY to enable tracing")

# Init Redis
r = redis.from_url(config.redis.url, decode_responses=True)
r_binary = redis.from_url(config.redis.url, decode_responses=False)

# Init embedder (runs on laptop CPU/GPU)
print("Loading MobileNetV2 embedder...")
embedder = Embedder()
print(f"Embedder loaded (dim={embedder.dim}, device={embedder.device})")

# Init memory and gating
memory = VectorMemory(r_binary, embedding_dim=embedder.dim)
gating = GatingLogic(
    known_threshold=config.perception.known_threshold,
    unknown_threshold=config.perception.unknown_threshold,
    margin_threshold=config.perception.margin_threshold,
)

# Track state (for cooldown and persistence tracking across frames)
tracker = SimpleTracker(iou_threshold=0.3, max_age=5.0)

app = FastAPI(title="OpenClawdIRL API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Weave-instrumented ops ---

@weave.op()
def embed_crop_op(crop_array: np.ndarray) -> np.ndarray:
    """Embed an object crop."""
    return embedder.embed(crop_array)


@weave.op()
def knn_lookup_op(vector: np.ndarray) -> list[dict]:
    """KNN lookup against vector memory."""
    return memory.knn_lookup(vector, k=5)


@weave.op()
def gate_decision_op(
    matches: list[dict],
    yolo_class: str,
    yolo_confidence: float,
    depth_value: float = None,
    crop_quality: float = None,
    prefilter_action: str = None,
) -> dict:
    """Decide known/uncertain/unknown using KNN + YOLO + depth signals."""
    state, label, similarity = gating.decide(matches)

    # If gating returned None for label, use YOLO class
    # (happens for "uncertain" and "unknown" states)
    if label is None:
        label = yolo_class

    # If KNN has no opinion but YOLO is very confident, trust YOLO
    if state == "unknown" and yolo_confidence >= 0.75:
        state = "uncertain"
        label = yolo_class
        similarity = yolo_confidence

    # Priority learning target: Jetson flagged this as low-confidence but
    # high-quality capture — promote to uncertain so the learning flow
    # engages sooner (research/teach).
    if prefilter_action == "priority" and state == "unknown":
        state = "uncertain"
        label = yolo_class

    # Depth-based confidence adjustments (only when depth is available)
    if depth_value is not None:
        # Close object + borderline uncertain → trust KNN more (good crop)
        if depth_value > 0.5 and state == "uncertain" and similarity >= 0.6:
            state = "known"
        # Far object + borderline known → trust KNN less (poor crop quality)
        if depth_value < 0.15 and state == "known" and similarity < 0.8:
            state = "uncertain"

    return {
        "state": state,
        "label": label,
        "similarity": similarity,
        "yolo_class": yolo_class,
        "yolo_confidence": yolo_confidence,
        "depth_value": depth_value,
        "crop_quality": crop_quality,
    }


@weave.op()
def learn_label_op(label_name: str, vectors: list) -> str:
    """Store embeddings under a label in memory."""
    return memory.learn_label(label_name, vectors)


# --- Request/Response Models ---

class Detection(BaseModel):
    bbox: list[int]
    yolo_class: str
    yolo_confidence: float
    crop_b64: str
    depth_value: Optional[float] = None
    crop_quality: Optional[float] = None
    prefilter_action: Optional[str] = None


class IngestRequest(BaseModel):
    timestamp: float
    frame_id: int
    detections: list[Detection]


class LabelRequest(BaseModel):
    track_id: Optional[str] = None
    unknown_event_id: Optional[str] = None
    label_name: str


class RenameRequest(BaseModel):
    label_id: str
    new_name: str


# --- Helper to decode crop ---

def decode_crop(crop_b64: str) -> np.ndarray:
    """Decode base64 JPEG to numpy array (BGR)."""
    img_bytes = base64.b64decode(crop_b64)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    arr = np.array(pil_img)
    # Convert RGB to BGR for consistency
    return arr[:, :, ::-1].copy()


# --- Endpoints ---

@app.get("/health")
def health():
    try:
        r.ping()
        return {"status": "ok", "redis": "connected", "embedder": "loaded"}
    except Exception as e:
        return {"status": "error", "redis": str(e)}


@weave.op()
@app.post("/ingest")
def ingest_frame(req: IngestRequest, background_tasks: BackgroundTasks):
    """Receive detections + crops from Jetson, run embedding + KNN + gating.

    This is the main endpoint called by the Jetson client every frame.
    """
    now = time.time()

    # Convert detections to tracker format
    det_list = [{"bbox": d.bbox, "class_name": d.yolo_class} for d in req.detections]
    tracks = tracker.update(det_list)

    # Build a map from bbox to detection for matching
    results = []

    for det in req.detections:
        # Find the matching track
        matched_track = None
        for track in tracks:
            if track.bbox == det.bbox:
                matched_track = track
                break

        if not matched_track:
            continue

        # Decode and embed the crop
        try:
            crop = decode_crop(det.crop_b64)
            embedding = embed_crop_op(crop)
        except Exception as e:
            continue

        # EMA stabilize track embedding
        if matched_track.embedding is not None:
            matched_track.embedding = 0.8 * matched_track.embedding + 0.2 * embedding
            norm = np.linalg.norm(matched_track.embedding)
            if norm > 0:
                matched_track.embedding = matched_track.embedding / norm
        else:
            matched_track.embedding = embedding

        # KNN lookup
        matches = knn_lookup_op(matched_track.embedding)

        # Gate decision (combines KNN similarity + YOLO confidence + depth)
        decision = gate_decision_op(
            matches, det.yolo_class, det.yolo_confidence,
            depth_value=det.depth_value,
            crop_quality=det.crop_quality,
            prefilter_action=det.prefilter_action,
        )
        state = decision["state"]
        label = decision["label"]
        similarity = decision["similarity"]

        matched_track.state = state
        matched_track.label = label
        matched_track.similarity = similarity

        # Publish object event to Redis stream
        event = ObjectEvent(
            track_id=matched_track.track_id,
            bbox=det.bbox,
            state=state,
            label=label,
            similarity=similarity,
            thumbnail_b64=det.crop_b64,
            embedding=matched_track.embedding.tolist() if state != "known" else None,
            depth_value=det.depth_value,
            crop_quality=det.crop_quality,
        )
        r.xadd(STREAM_VISION_OBJECTS, event.to_redis())

        # Update metrics
        r.incr(METRICS_TOTAL_QUERIES)
        if state == "known":
            r.incr(METRICS_KNOWN_COUNT)

        # Unknown gating with cooldown + persistence
        if state == "unknown" and matched_track.frames_seen >= config.perception.persistence_frames:
            if now > matched_track.cooldown_until:
                matched_track.cooldown_until = now + config.perception.cooldown_seconds
                unknown_event = UnknownEvent(
                    track_id=matched_track.track_id,
                    thumbnail_b64=det.crop_b64,
                    embedding=matched_track.embedding.tolist(),
                    top_similarity=similarity,
                    depth_value=det.depth_value,
                    crop_quality=det.crop_quality,
                    yolo_class=det.yolo_class,
                    yolo_confidence=det.yolo_confidence,
                )
                r.xadd(STREAM_VISION_UNKNOWN, unknown_event.to_redis())
                r.incr(METRICS_UNKNOWN_COUNT)
                # Auto-trigger image-based research via Claude/GPT vision
                background_tasks.add_task(
                    _do_research,
                    track_id=matched_track.track_id,
                    thumbnail_b64=det.crop_b64,
                    yolo_hint=det.yolo_class,
                    yolo_confidence=det.yolo_confidence,
                )

        results.append({
            "track_id": matched_track.track_id,
            "state": state,
            "label": label,
            "similarity": round(similarity, 4),
            "yolo_class": det.yolo_class,
            "yolo_confidence": det.yolo_confidence,
            "depth_value": det.depth_value,
            "crop_quality": det.crop_quality,
        })

    return {"objects": results, "track_count": len(tracks)}


@app.get("/events/objects")
def get_object_events(count: int = 50):
    """Get recent object events."""
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
    """Get recent unknown events."""
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
    """Teach the system a new label for an object."""
    embedding = None
    thumbnail = ""

    # Search unknown events first, then object events
    for stream in [STREAM_VISION_UNKNOWN, STREAM_VISION_OBJECTS]:
        entries = r.xrevrange(stream, count=200)
        for entry_id, data in entries:
            match = False
            if req.track_id and data.get("track_id") == req.track_id:
                match = True
            if req.unknown_event_id and data.get("event_id") == req.unknown_event_id:
                match = True
            if match:
                emb_str = data.get("embedding", "")
                if emb_str:
                    embedding = np.array(json.loads(emb_str), dtype=np.float32)
                thumbnail = data.get("thumbnail_b64", "")
                break
        if embedding is not None:
            break

    if embedding is None:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding found for track_id={req.track_id}",
        )

    # Store in memory
    label_id = learn_label_op(req.label_name, [embedding])

    # Emit label event
    label_event = LabelEvent(
        track_id=req.track_id or "",
        label_name=req.label_name,
        label_id=label_id,
    )
    r.xadd(STREAM_VISION_LABELS, label_event.to_redis())

    return {
        "label_id": label_id,
        "label_name": req.label_name,
        "message": f"Learned '{req.label_name}' with 1 exemplar",
    }


@weave.op()
@app.post("/research")
def trigger_research(track_id: str, background_tasks: BackgroundTasks):
    """Trigger Browserbase research for an unknown object."""
    # Find the unknown event
    entries = r.xrevrange(STREAM_VISION_UNKNOWN, count=100)
    target = None
    for entry_id, data in entries:
        if data.get("track_id") == track_id:
            target = data
            break

    if not target:
        raise HTTPException(status_code=404, detail=f"No unknown event for track {track_id}")

    # Queue research in background
    yolo_conf_str = target.get("yolo_confidence", "0")
    yolo_conf = float(yolo_conf_str) if yolo_conf_str else 0.0

    background_tasks.add_task(
        _do_research,
        track_id=track_id,
        thumbnail_b64=target.get("thumbnail_b64", ""),
        yolo_hint=target.get("yolo_class", ""),
        yolo_confidence=yolo_conf,
    )

    return {"status": "research_queued", "track_id": track_id}


def _do_research(track_id: str, thumbnail_b64: str, yolo_hint: str,
                  yolo_confidence: float = 0.0):
    """Background task: research an unknown object via vision APIs (Claude/GPT)."""
    try:
        from workers.research.researcher import research_object
        result = research_object(thumbnail_b64, yolo_hint, yolo_confidence=yolo_confidence)

        if result and result.get("confidence", 0) > 0.7:
            # Auto-label if research is confident
            # Find embedding for this track
            entries = r.xrevrange(STREAM_VISION_UNKNOWN, count=100)
            for entry_id, data in entries:
                if data.get("track_id") == track_id:
                    emb_str = data.get("embedding", "")
                    if emb_str:
                        embedding = np.array(json.loads(emb_str), dtype=np.float32)
                        label_name = result["label"]
                        label_id = learn_label_op(label_name, [embedding])

                        # Emit events
                        label_event = LabelEvent(
                            track_id=track_id,
                            label_name=label_name,
                            label_id=label_id,
                        )
                        r.xadd(STREAM_VISION_LABELS, label_event.to_redis())

                        # Store research result
                        r.xadd(STREAM_VISION_RESEARCHED, {
                            "track_id": track_id,
                            "label": label_name,
                            "confidence": str(result["confidence"]),
                            "description": result.get("description", ""),
                            "source": result.get("source", ""),
                            "auto_labeled": "true",
                            "timestamp": str(time.time()),
                        })
                    break
        else:
            # Low confidence — put in queue for manual review
            r.xadd(STREAM_VISION_RESEARCHED, {
                "track_id": track_id,
                "label": result.get("label", "unknown") if result else "unknown",
                "confidence": str(result.get("confidence", 0)) if result else "0",
                "description": result.get("description", "") if result else "",
                "auto_labeled": "false",
                "timestamp": str(time.time()),
            })

    except ImportError:
        print(f"Research worker not available for track {track_id}")
    except Exception as e:
        print(f"Research failed for track {track_id}: {e}")


@app.get("/memory")
def get_memory():
    """Get all learned labels."""
    labels = memory.get_all_labels()
    total_embeddings = memory.get_memory_count()
    return {"labels": labels, "total_embeddings": total_embeddings}


@app.post("/rename")
def rename_label(req: RenameRequest):
    """Rename a label."""
    lkey = label_key(req.label_id)
    if not r.exists(lkey):
        raise HTTPException(status_code=404, detail=f"Label {req.label_id} not found")

    r.hset(lkey, "name", req.new_name)

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

    recognition_rate = known_count / total_queries if total_queries > 0 else 0.0
    unknown_rate = unknown_count / total_queries if total_queries > 0 else 0.0

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
    """Get label events over time."""
    entries = r.xrange(STREAM_VISION_LABELS, count=100)
    history = []
    for entry_id, data in entries:
        history.append({
            "timestamp": float(data.get("timestamp", 0)),
            "label_name": data.get("label_name", ""),
            "event": "label_added",
        })
    return {"history": history}


# Camera feed management
camera_process = None

@app.post("/camera/start")
def start_camera(camera_id: int = 0, fps: float = 2.0, confidence: float = 0.15):
    """Start camera feed."""
    import subprocess
    global camera_process

    if camera_process and camera_process.poll() is None:
        return {"status": "already_running", "pid": camera_process.pid}

    # Start jetson_client as subprocess
    cmd = [
        sys.executable,
        "perception/jetson_client.py",
        "--fallback",
        "--no-depth",
        "--camera", str(camera_id),
        "--fps", str(fps),
        "--confidence", str(confidence),
        "--backend", "http://localhost:8003",
    ]

    camera_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(os.path.dirname(__file__))
    )

    return {"status": "started", "pid": camera_process.pid, "camera_id": camera_id}


@app.post("/camera/stop")
def stop_camera():
    """Stop camera feed."""
    global camera_process

    if camera_process and camera_process.poll() is None:
        camera_process.terminate()
        camera_process.wait(timeout=5)
        camera_process = None
        return {"status": "stopped"}

    return {"status": "not_running"}


@app.get("/camera/status")
def camera_status():
    """Get camera feed status."""
    global camera_process

    if camera_process and camera_process.poll() is None:
        return {"running": True, "pid": camera_process.pid}

    return {"running": False}


@app.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    """Stream real-time detection results to frontend for live overlay."""
    import asyncio

    await websocket.accept()

    # Track last seen message ID for each stream
    last_id = ["0-0"]  # Use list for mutability in closure

    def read_stream():
        """Blocking Redis read in thread."""
        return r.xread({STREAM_VISION_OBJECTS: last_id[0]}, count=10, block=100)

    try:
        while True:
            # Read new detections from the object stream (in thread to avoid blocking)
            entries = await asyncio.to_thread(read_stream)

            if entries:
                for stream_name, messages in entries:
                    for msg_id, data in messages:
                        last_id[0] = msg_id

                        # Send detection data to frontend
                        await websocket.send_json({
                            "track_id": data.get(b"track_id", b"").decode() if isinstance(data.get(b"track_id"), bytes) else data.get("track_id", ""),
                            "state": data.get(b"state", b"").decode() if isinstance(data.get(b"state"), bytes) else data.get("state", ""),
                            "label": data.get(b"label", b"").decode() if isinstance(data.get(b"label"), bytes) else data.get("label", ""),
                            "bbox": data.get(b"bbox", b"").decode() if isinstance(data.get(b"bbox"), bytes) else data.get("bbox", ""),
                            "similarity": float(data.get(b"similarity", b"0").decode() if isinstance(data.get(b"similarity"), bytes) else data.get("similarity", "0")),
                            "timestamp": float(data.get(b"timestamp", b"0").decode() if isinstance(data.get(b"timestamp"), bytes) else data.get("timestamp", "0")),
                        })
    except WebSocketDisconnect:
        pass


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
