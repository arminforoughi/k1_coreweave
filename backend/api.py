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
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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

# Ensure vector index exists (auto-create if missing, e.g. after FLUSHALL)
try:
    r.execute_command("FT.INFO", "idx:embeddings")
except Exception:
    print("Vector index missing — creating idx:embeddings...")
    r.execute_command(
        "FT.CREATE", "idx:embeddings",
        "ON", "JSON",
        "PREFIX", "1", "emb:",
        "SCHEMA",
        "$.label_id", "AS", "label_id", "TAG",
        "$.label_name", "AS", "label_name", "TEXT",
        "$.created_at", "AS", "created_at", "NUMERIC",
        "$.vector", "AS", "vector", "VECTOR", "FLAT", "6",
        "TYPE", "FLOAT32", "DIM", str(embedder.dim), "DISTANCE_METRIC", "COSINE",
    )
    print("Vector index created.")

# Init memory and gating
memory = VectorMemory(r_binary, embedding_dim=embedder.dim)
gating = GatingLogic(
    known_threshold=config.perception.known_threshold,
    unknown_threshold=config.perception.unknown_threshold,
    margin_threshold=config.perception.margin_threshold,
)

# Track state (for cooldown and persistence tracking across frames)
tracker = SimpleTracker(iou_threshold=0.2, max_age=10.0)

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
    """Gate detections: KNN determines state, YOLO provides a hint.

    Logic:
    - If the object is NOT in the KNN → it's "unknown" regardless of YOLO.
      YOLO's class is just a hint passed to research for context.
    - If the object IS in the KNN with a strong match → "known" with the
      learned label. The system has verified this object before.
    - "uncertain" = KNN has a moderate match but not confident enough.
    - This ensures every new object triggers research, even if YOLO
      confidently misclassifies it (e.g., robot → "airplane").
    """
    # Look at raw KNN matches directly — top match similarity and label
    if matches:
        knn_similarity = matches[0]["similarity"]
        knn_label = matches[0]["label_name"]
    else:
        knn_similarity = 0.0
        knn_label = None

    # KNN determines the state — it represents our verified knowledge.
    # If the KNN has a strong match (similarity >= 0.7), the object is known
    # regardless of margin — we've seen something very similar before.
    # Otherwise, the object is unknown and needs research.
    if knn_label and knn_similarity >= 0.7:
        state = "known"
        label = knn_label
        confidence = knn_similarity
    elif knn_label and knn_similarity >= 0.5:
        state = "uncertain"
        label = knn_label
        confidence = knn_similarity
    else:
        # Not in KNN or weak match → unknown, use YOLO class as display hint
        state = "unknown"
        label = yolo_class
        confidence = yolo_confidence

    return {
        "state": state,
        "label": label,
        "similarity": confidence,
        "yolo_class": yolo_class,
        "yolo_confidence": yolo_confidence,
        "knn_label": knn_label,
        "knn_similarity": knn_similarity,
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
    frame_b64: Optional[str] = None


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

    # Store latest frame for MJPEG streaming
    if req.frame_b64:
        r.set("latest_frame", req.frame_b64)

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
        # Include embedding for unknown objects (needed for research/learning)
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
            yolo_class=det.yolo_class,
            yolo_confidence=det.yolo_confidence,
        )
        r.xadd(STREAM_VISION_OBJECTS, event.to_redis())

        # Update metrics
        r.incr(METRICS_TOTAL_QUERIES)
        if state == "known":
            r.incr(METRICS_KNOWN_COUNT)

        # Research trigger: anything NOT confidently known should be researched.
        # "unknown" = no KNN match, "uncertain" = weak KNN match (unreliable).
        # Both need research — e.g. a tissue box getting 60% sim to a Spot robot
        # doesn't mean it IS a Spot. Research will confirm or identify the real object.
        if state in ("unknown", "uncertain") and matched_track.frames_seen >= config.perception.persistence_frames:
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
            "bbox": det.bbox,
            "yolo_class": det.yolo_class,
            "yolo_confidence": det.yolo_confidence,
            "depth_value": det.depth_value,
            "crop_quality": det.crop_quality,
        })

    # Store latest detections for MJPEG overlay
    if results:
        r.set("latest_detections", json.dumps(results))

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


@app.get("/events/researched")
def get_researched_events(count: int = 50):
    """Get recent research results with enrichment data."""
    try:
        entries = r.xrevrange(STREAM_VISION_RESEARCHED, count=count)
        events = []
        for entry_id, data in entries:
            data["stream_id"] = entry_id
            # Parse JSON fields back
            if data.get("specs"):
                try:
                    data["specs"] = json.loads(data["specs"])
                except (json.JSONDecodeError, TypeError):
                    pass
            if data.get("facts"):
                try:
                    data["facts"] = json.loads(data["facts"])
                except (json.JSONDecodeError, TypeError):
                    pass
            if data.get("search_sources"):
                try:
                    data["search_sources"] = json.loads(data["search_sources"])
                except (json.JSONDecodeError, TypeError):
                    pass
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

                        # Store research result with enrichment data
                        research_data = {
                            "track_id": track_id,
                            "label": label_name,
                            "confidence": str(result["confidence"]),
                            "description": result.get("description", ""),
                            "source": result.get("source", ""),
                            "auto_labeled": "true",
                            "timestamp": str(time.time()),
                            "thumbnail_b64": thumbnail_b64,
                        }
                        if result.get("facts"):
                            research_data["facts"] = json.dumps(result["facts"])
                        # Add Browserbase enrichment fields if present
                        if result.get("manufacturer"):
                            research_data["manufacturer"] = result["manufacturer"]
                        if result.get("price"):
                            research_data["price"] = result["price"]
                        if result.get("specs"):
                            research_data["specs"] = json.dumps(result["specs"])
                        if result.get("safety_info"):
                            research_data["safety_info"] = result["safety_info"]
                        if result.get("web_description"):
                            research_data["web_description"] = result["web_description"]
                        if result.get("product_url"):
                            research_data["product_url"] = result["product_url"]
                        if result.get("search_sources"):
                            research_data["search_sources"] = json.dumps(result["search_sources"])
                        r.xadd(STREAM_VISION_RESEARCHED, research_data)
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
                "thumbnail_b64": thumbnail_b64,
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


# --- Live video stream ---

def _draw_overlays(frame_bytes: bytes) -> bytes:
    """Draw bounding box overlays on a JPEG frame. Returns new JPEG bytes."""
    import cv2

    det_json = r.get("latest_detections")
    if not det_json:
        return frame_bytes

    try:
        detections = json.loads(det_json)
    except Exception:
        return frame_bytes

    # Decode JPEG to numpy array
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return frame_bytes

    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        state = det.get("state", "unknown")
        label = det.get("label", state)
        similarity = det.get("similarity", 0)

        # Color based on state: green=known, yellow=uncertain, red=unknown
        if state == "known":
            color = (0, 255, 0)
        elif state == "uncertain":
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Show confidence (YOLO primary, or KNN if learned label)
        conf_val = det.get("similarity", 0)
        label_text = f"{label} ({int(conf_val * 100)}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), thickness)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


@app.get("/stream/mjpeg")
def mjpeg_stream():
    """MJPEG stream of the Jetson camera with bounding box overlays."""
    def generate():
        while True:
            frame_b64 = r.get("latest_frame")
            if frame_b64:
                if isinstance(frame_b64, bytes):
                    frame_b64 = frame_b64.decode("utf-8")
                frame_bytes = base64.b64decode(frame_b64)
                frame_bytes = _draw_overlays(frame_bytes)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
            time.sleep(0.15)  # ~6-7 fps output

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/snapshot")
def snapshot():
    """Single JPEG snapshot of the latest Jetson camera frame with overlays."""
    frame_b64 = r.get("latest_frame")
    if not frame_b64:
        raise HTTPException(status_code=404, detail="No frame available")
    if isinstance(frame_b64, bytes):
        frame_b64 = frame_b64.decode("utf-8")
    frame_bytes = base64.b64decode(frame_b64)
    frame_bytes = _draw_overlays(frame_bytes)
    return StreamingResponse(
        BytesIO(frame_bytes),
        media_type="image/jpeg",
    )


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


# --- Voice Chat Endpoints ---

class VoiceQueryRequest(BaseModel):
    """Voice query with optional audio data."""
    text: Optional[str] = None  # Transcribed text (if client transcribes)
    audio_b64: Optional[str] = None  # Raw audio as base64 (if server transcribes)


@app.post("/voice/query")
async def voice_query(audio: Optional[UploadFile] = None, text: Optional[str] = None):
    """Process voice query about what the robot sees."""
    import openai
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    client = openai.OpenAI(api_key=openai_key)
    
    # Get user text (either from transcription or direct text)
    user_text = text
    if audio and not user_text:
        # Transcribe audio
        audio_data = await audio.read()
        audio_file = BytesIO(audio_data)
        audio_file.name = "recording.webm"
        
        try:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            user_text = transcript.text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    
    if not user_text:
        raise HTTPException(status_code=400, detail="No text or audio provided")
    
    # Get vision context from Redis
    frame_b64 = r.get("latest_frame")
    det_json = r.get("latest_detections")
    
    detections = []
    if det_json:
        try:
            detections = json.loads(det_json)
        except:
            pass
    
    # Format detection summary
    if detections:
        counts = {}
        for det in detections:
            cls = det.get("label", det.get("yolo_class", "unknown"))
            counts[cls] = counts.get(cls, 0) + 1
        detection_summary = "Currently detecting: " + ", ".join(
            f"{count} {name}{'s' if count > 1 else ''}" 
            for name, count in counts.items()
        )
    else:
        detection_summary = "No objects currently detected"
    
    # Build prompt
    system_prompt = """You are K1, a friendly robot assistant with vision capabilities.
You can see through your camera and describe what you observe.

When asked about what you see:
1. Describe the scene naturally and conversationally
2. Mention specific objects with their positions
3. Be helpful but keep responses concise (2-3 sentences)

Keep responses short and conversational - you're speaking out loud!"""

    # Build message with image if available
    if frame_b64:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"[{detection_summary}]\n\nUser: {user_text}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_b64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[{detection_summary}]\n\nUser: {user_text}"}
        ]
    
    # Get LLM response
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}")
    
    return {
        "transcription": user_text,
        "response": response_text,
        "detections": detection_summary
    }


@app.get("/voice/speak")
async def voice_speak(text: str):
    """Convert text to speech."""
    import openai
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    client = openai.OpenAI(api_key=openai_key)
    
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        
        return StreamingResponse(
            BytesIO(response.content),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


# --- Manual Video Pipeline (SSE) ---

@app.post("/pipeline/run")
async def run_pipeline(video: UploadFile):
    """Process an uploaded video through the full pipeline, streaming SSE progress.

    Each step emits a Server-Sent Event so the dashboard can render a live
    play-by-play terminal feed.
    """
    import cv2
    import tempfile
    from ultralytics import YOLO

    # Save upload to temp file so cv2 can read it
    tmp = tempfile.NamedTemporaryFile(suffix=".mov", delete=False)
    content = await video.read()
    tmp.write(content)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    def generate():
        """Generator that yields SSE events as the pipeline runs."""
        yolo_model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            yield _sse({"type": "error", "message": "Failed to open video file"})
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_fps = 2.0
        frame_skip = max(1, int(fps / target_fps))

        yield _sse({"type": "info", "message": f"Video loaded: {total_frames} frames at {fps:.0f} FPS, sampling every {frame_skip} frames"})
        yield _sse({"type": "info", "message": f"KNN memory: {len(memory.get_all_labels())} labels, {memory.get_memory_count()} embeddings"})

        # Local tracker for this run
        local_tracker = SimpleTracker(iou_threshold=0.2, max_age=10.0)
        frame_idx = 0
        processed = 0
        research_queue = []  # Collect research tasks to run after detection pass

        prefilter_cfg = {
            "depth_min": 0.08, "depth_max": 0.95,
            "crop_quality_min": 0.15, "confidence_low": 0.5,
            "confidence_floor": 0.25,
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue
            processed += 1
            video_ts = frame_idx / fps

            # YOLO detection
            results = yolo_model(frame, conf=0.25, verbose=False)
            from perception.jetson_client import process_yolo_results, encode_crop
            detections, filtered = process_yolo_results(
                results, frame, None, None, prefilter_cfg
            )

            if not detections:
                continue

            yield _sse({"type": "frame", "frame": processed, "time": round(video_ts, 1),
                        "detections": len(detections), "filtered": filtered})

            # Convert to tracker format
            det_list = [{"bbox": d["bbox"], "class_name": d["yolo_class"]} for d in detections]
            tracks = local_tracker.update(det_list)

            # Store full frame for MJPEG preview
            h, w = frame.shape[:2]
            scale = min(640 / max(h, w), 1.0)
            preview = cv2.resize(frame, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buf).decode("ascii")
            r.set("latest_frame", frame_b64)

            for det in detections:
                # Match detection to track
                matched_track = None
                for track in tracks:
                    if track.bbox == det["bbox"]:
                        matched_track = track
                        break
                if not matched_track:
                    continue

                # Embed
                try:
                    crop = decode_crop(det["crop_b64"])
                    embedding = embed_crop_op(crop)
                except Exception:
                    continue

                # EMA
                if matched_track.embedding is not None:
                    matched_track.embedding = 0.8 * matched_track.embedding + 0.2 * embedding
                    norm = np.linalg.norm(matched_track.embedding)
                    if norm > 0:
                        matched_track.embedding = matched_track.embedding / norm
                else:
                    matched_track.embedding = embedding

                # KNN
                matches = knn_lookup_op(matched_track.embedding)
                decision = gate_decision_op(
                    matches, det["yolo_class"], det["yolo_confidence"],
                    depth_value=det.get("depth_value"),
                    crop_quality=det.get("crop_quality"),
                )
                state = decision["state"]
                label = decision["label"]
                similarity = decision["similarity"]

                matched_track.state = state
                matched_track.label = label
                matched_track.similarity = similarity

                # Publish to Redis streams (so dashboard tabs work too)
                event = ObjectEvent(
                    track_id=matched_track.track_id,
                    bbox=det["bbox"],
                    state=state,
                    label=label,
                    similarity=similarity,
                    thumbnail_b64=det["crop_b64"],
                    embedding=matched_track.embedding.tolist() if state != "known" else None,
                    depth_value=det.get("depth_value"),
                    crop_quality=det.get("crop_quality"),
                    yolo_class=det["yolo_class"],
                    yolo_confidence=det["yolo_confidence"],
                )
                r.xadd(STREAM_VISION_OBJECTS, event.to_redis())
                r.incr(METRICS_TOTAL_QUERIES)
                if state == "known":
                    r.incr(METRICS_KNOWN_COUNT)

                knn_info = f"KNN: {decision['knn_label'] or 'no match'} ({decision['knn_similarity']:.0%})" if decision['knn_similarity'] > 0 else "KNN: no match"

                yield _sse({
                    "type": "detection",
                    "track_id": matched_track.track_id,
                    "yolo_class": det["yolo_class"],
                    "yolo_conf": round(det["yolo_confidence"] * 100, 1),
                    "state": state,
                    "label": label,
                    "similarity": round(similarity * 100, 1),
                    "frames_seen": matched_track.frames_seen,
                    "knn_info": knn_info,
                })

                # Research trigger
                now = time.time()
                if state in ("unknown", "uncertain") and matched_track.frames_seen >= config.perception.persistence_frames:
                    if now > matched_track.cooldown_until:
                        matched_track.cooldown_until = now + config.perception.cooldown_seconds

                        unknown_event = UnknownEvent(
                            track_id=matched_track.track_id,
                            thumbnail_b64=det["crop_b64"],
                            embedding=matched_track.embedding.tolist(),
                            top_similarity=similarity,
                            depth_value=det.get("depth_value"),
                            crop_quality=det.get("crop_quality"),
                            yolo_class=det["yolo_class"],
                            yolo_confidence=det["yolo_confidence"],
                        )
                        r.xadd(STREAM_VISION_UNKNOWN, unknown_event.to_redis())
                        r.incr(METRICS_UNKNOWN_COUNT)

                        yield _sse({
                            "type": "persistence",
                            "track_id": matched_track.track_id,
                            "yolo_class": det["yolo_class"],
                            "frames_seen": matched_track.frames_seen,
                            "message": f"Persistence threshold reached ({matched_track.frames_seen}/{config.perception.persistence_frames}) — research triggered!",
                        })

                        research_queue.append({
                            "track_id": matched_track.track_id,
                            "thumbnail_b64": det["crop_b64"],
                            "yolo_hint": det["yolo_class"],
                            "yolo_confidence": det["yolo_confidence"],
                            "embedding": matched_track.embedding.tolist(),
                        })

                # Store latest detections for overlay
                det_result = {
                    "track_id": matched_track.track_id,
                    "state": state, "label": label,
                    "similarity": round(similarity, 4),
                    "bbox": det["bbox"],
                    "yolo_class": det["yolo_class"],
                    "yolo_confidence": det["yolo_confidence"],
                }
                r.set("latest_detections", json.dumps([det_result]))

        cap.release()
        os.unlink(tmp_path)

        yield _sse({"type": "info", "message": f"Video scan complete. {processed} frames processed, {len(research_queue)} objects queued for research."})

        # Now run research synchronously so we can stream progress
        if research_queue:
            yield _sse({"type": "info", "message": "Starting research phase..."})

            from workers.research.researcher import research_object
            for i, item in enumerate(research_queue):
                track_id = item["track_id"]
                yield _sse({
                    "type": "research_start",
                    "track_id": track_id,
                    "yolo_class": item["yolo_hint"],
                    "index": i + 1,
                    "total": len(research_queue),
                    "message": f"Researching object {i+1}/{len(research_queue)}: {item['yolo_hint']}...",
                })

                try:
                    result = research_object(
                        item["thumbnail_b64"],
                        item["yolo_hint"],
                        yolo_confidence=item["yolo_confidence"],
                    )

                    if result and result.get("confidence", 0) > 0.7:
                        label_name = result["label"]
                        embedding = np.array(item["embedding"], dtype=np.float32)
                        label_id = learn_label_op(label_name, [embedding])

                        label_event = LabelEvent(
                            track_id=track_id,
                            label_name=label_name,
                            label_id=label_id,
                        )
                        r.xadd(STREAM_VISION_LABELS, label_event.to_redis())

                        research_data = {
                            "track_id": track_id,
                            "label": label_name,
                            "confidence": str(result["confidence"]),
                            "description": result.get("description", ""),
                            "source": result.get("source", ""),
                            "auto_labeled": "true",
                            "timestamp": str(time.time()),
                            "thumbnail_b64": item["thumbnail_b64"],
                        }
                        if result.get("facts"):
                            research_data["facts"] = json.dumps(result["facts"])
                        if result.get("manufacturer"):
                            research_data["manufacturer"] = result["manufacturer"]
                        if result.get("price"):
                            research_data["price"] = result["price"]
                        if result.get("specs"):
                            research_data["specs"] = json.dumps(result["specs"])
                        if result.get("safety_info"):
                            research_data["safety_info"] = result["safety_info"]
                        if result.get("web_description"):
                            research_data["web_description"] = result["web_description"]
                        if result.get("product_url"):
                            research_data["product_url"] = result["product_url"]
                        if result.get("search_sources"):
                            research_data["search_sources"] = json.dumps(result["search_sources"])
                        r.xadd(STREAM_VISION_RESEARCHED, research_data)

                        yield _sse({
                            "type": "research_complete",
                            "track_id": track_id,
                            "label": label_name,
                            "confidence": round(result["confidence"] * 100, 1),
                            "source": result.get("source", ""),
                            "description": result.get("description", ""),
                            "manufacturer": result.get("manufacturer"),
                            "price": result.get("price"),
                            "specs_count": len(result.get("specs", [])),
                            "has_safety": bool(result.get("safety_info")),
                            "message": f"Learned: {label_name} ({result['confidence']:.0%} confidence)",
                        })
                    else:
                        conf = result.get("confidence", 0) if result else 0
                        lbl = result.get("label", "unknown") if result else "unknown"
                        yield _sse({
                            "type": "research_low_conf",
                            "track_id": track_id,
                            "label": lbl,
                            "confidence": round(conf * 100, 1),
                            "message": f"Low confidence: {lbl} ({conf:.0%}) — queued for review",
                        })

                except Exception as e:
                    yield _sse({
                        "type": "research_error",
                        "track_id": track_id,
                        "message": f"Research failed: {str(e)[:100]}",
                    })

        # Final summary
        labels_now = memory.get_all_labels()
        emb_count = memory.get_memory_count()
        unknown_count = int(r.get(METRICS_UNKNOWN_COUNT) or 0)
        known_count = int(r.get(METRICS_KNOWN_COUNT) or 0)
        total_queries = int(r.get(METRICS_TOTAL_QUERIES) or 0)

        yield _sse({
            "type": "complete",
            "labels": len(labels_now),
            "embeddings": emb_count,
            "unknown_count": unknown_count,
            "known_count": known_count,
            "total_queries": total_queries,
            "message": f"Pipeline complete. {len(labels_now)} objects learned, {emb_count} embeddings stored.",
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/pipeline/reset")
def pipeline_reset():
    """Flush all learned data for a fresh demo run."""
    for pattern in ["emb:*", "label:*", "track:*", "objcard:*"]:
        keys = list(r.scan_iter(pattern))
        if keys:
            r.delete(*keys)
    for stream in [STREAM_VISION_OBJECTS, STREAM_VISION_UNKNOWN,
                   STREAM_VISION_LABELS, STREAM_VISION_RESEARCHED]:
        try:
            r.delete(stream)
        except Exception:
            pass
    for k in [METRICS_UNKNOWN_COUNT, METRICS_KNOWN_COUNT, METRICS_TOTAL_QUERIES]:
        r.delete(k)
    r.delete("latest_frame", "latest_detections")
    try:
        r.execute_command("FT.DROPINDEX", "idx:embeddings", "DD")
    except Exception:
        pass
    # Recreate vector index
    r.execute_command(
        "FT.CREATE", "idx:embeddings",
        "ON", "JSON",
        "PREFIX", "1", "emb:",
        "SCHEMA",
        "$.label_id", "AS", "label_id", "TAG",
        "$.label_name", "AS", "label_name", "TEXT",
        "$.created_at", "AS", "created_at", "NUMERIC",
        "$.vector", "AS", "vector", "VECTOR", "FLAT", "6",
        "TYPE", "FLOAT32", "DIM", str(embedder.dim), "DISTANCE_METRIC", "COSINE",
    )
    # Reset tracker
    global tracker
    tracker = SimpleTracker(iou_threshold=0.2, max_age=10.0)
    return {"status": "reset", "message": "All data flushed. Ready for fresh run."}


def _sse(data: dict) -> str:
    """Format a dict as an SSE event."""
    return f"data: {json.dumps(data)}\n\n"


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
