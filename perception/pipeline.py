"""Main perception pipeline: camera -> detect -> track -> embed -> classify -> publish."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import json
import base64
import cv2
import numpy as np
import redis
import weave
from dotenv import load_dotenv

from shared.config import load_config
from shared.redis_keys import (
    STREAM_VISION_OBJECTS,
    STREAM_VISION_UNKNOWN,
    METRICS_UNKNOWN_COUNT,
    METRICS_KNOWN_COUNT,
    METRICS_TOTAL_QUERIES,
    track_key,
)
from shared.events import ObjectEvent, UnknownEvent
from perception.detector import Detector
from perception.tracker import SimpleTracker
from perception.embedder import Embedder
from perception.memory import VectorMemory, GatingLogic


EMA_ALPHA = 0.2  # for stabilizing track embeddings


def crop_object(frame: np.ndarray, bbox: list) -> np.ndarray:
    """Crop an object from frame given bbox [x1, y1, x2, y2]."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return frame[y1:y2, x1:x2]


def encode_thumbnail(crop: np.ndarray, max_size: int = 128) -> str:
    """Encode crop as small base64 JPEG."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return ""
    scale = min(max_size / max(h, w), 1.0)
    resized = cv2.resize(crop, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode("ascii")


@weave.op()
def detect_objects(detector: Detector, frame: np.ndarray) -> list[dict]:
    return detector.detect(frame)


@weave.op()
def track_objects(tracker: SimpleTracker, detections: list[dict]):
    return tracker.update(detections)


@weave.op()
def embed_crop(embedder: Embedder, crop: np.ndarray) -> np.ndarray:
    return embedder.embed(crop)


@weave.op()
def knn_lookup(memory: VectorMemory, vector: np.ndarray) -> list[dict]:
    return memory.knn_lookup(vector, k=5)


@weave.op()
def gate_decision(gating: GatingLogic, matches: list[dict]) -> tuple:
    return gating.decide(matches)


def run_pipeline():
    """Main perception loop."""
    load_dotenv()
    config = load_config()

    # Init Weave
    weave.init(config.weave.project)

    # Init Redis
    r = redis.from_url(config.redis.url, decode_responses=True)
    r_binary = redis.from_url(config.redis.url, decode_responses=False)
    r.ping()
    print("Redis connected")

    # Init components
    print("Loading detector...")
    detector = Detector(confidence=config.perception.detection_confidence)
    print("Loading tracker...")
    tracker = SimpleTracker()
    print("Loading embedder...")
    embedder = Embedder()
    print("All models loaded")

    # Init memory and gating
    memory = VectorMemory(r_binary, embedding_dim=embedder.dim)
    gating = GatingLogic(
        known_threshold=config.perception.known_threshold,
        unknown_threshold=config.perception.unknown_threshold,
        margin_threshold=config.perception.margin_threshold,
    )

    # Open camera
    cap = cv2.VideoCapture(config.perception.camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {config.perception.camera_index}")
        return

    print(f"Camera {config.perception.camera_index} opened. Starting perception loop...")

    frame_count = 0
    last_publish = {}  # track_id -> last publish time (throttle to ~3 Hz)
    PUBLISH_INTERVAL = 0.33  # seconds

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed, retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1

            # Detect objects
            detections = detect_objects(detector, frame)

            # Track objects
            tracks = track_objects(tracker, detections)

            now = time.time()

            for track in tracks:
                # Skip if recently published
                last_pub = last_publish.get(track.track_id, 0)
                if now - last_pub < PUBLISH_INTERVAL:
                    continue

                # Crop and embed
                crop = crop_object(frame, track.bbox)
                if crop.size == 0:
                    continue

                e_frame = embed_crop(embedder, crop)

                # EMA stabilize track embedding
                if track.embedding is not None:
                    track.embedding = (1 - EMA_ALPHA) * track.embedding + EMA_ALPHA * e_frame
                    # Re-normalize
                    norm = np.linalg.norm(track.embedding)
                    if norm > 0:
                        track.embedding = track.embedding / norm
                else:
                    track.embedding = e_frame

                # KNN lookup
                matches = knn_lookup(memory, track.embedding)

                # Gate decision
                state, label, similarity = gate_decision(gating, matches)
                track.state = state
                track.label = label
                track.similarity = similarity

                # Encode thumbnail
                thumb = encode_thumbnail(crop)

                # Publish object event (include embedding for teach action)
                event = ObjectEvent(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    state=state,
                    label=label,
                    similarity=similarity,
                    thumbnail_b64=thumb,
                    embedding=track.embedding.tolist() if state != "known" else None,
                )
                r.xadd(STREAM_VISION_OBJECTS, event.to_redis(), maxlen=1000)
                last_publish[track.track_id] = now

                # Update metrics
                r.incr(METRICS_TOTAL_QUERIES)
                if state == "known":
                    r.incr(METRICS_KNOWN_COUNT)

                # Unknown gating with cooldown and persistence
                if state == "unknown" and track.frames_seen >= config.perception.persistence_frames:
                    if now > track.cooldown_until:
                        track.cooldown_until = now + config.perception.cooldown_seconds

                        unknown_event = UnknownEvent(
                            track_id=track.track_id,
                            thumbnail_b64=thumb,
                            embedding=track.embedding.tolist(),
                            top_similarity=similarity,
                        )
                        r.xadd(STREAM_VISION_UNKNOWN, unknown_event.to_redis(), maxlen=500)
                        r.incr(METRICS_UNKNOWN_COUNT)

                        print(f"[UNKNOWN] track={track.track_id} sim={similarity:.3f}")

                elif state == "known":
                    print(f"[KNOWN] track={track.track_id} label={label} sim={similarity:.3f}")

    except KeyboardInterrupt:
        print("\nStopping perception pipeline...")
    finally:
        cap.release()
        print("Camera released. Pipeline stopped.")


if __name__ == "__main__":
    run_pipeline()
