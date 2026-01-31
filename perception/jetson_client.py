"""Thin Jetson client: camera capture + YOLO detection + POST crops to backend.

This is the ONLY thing that runs on the Jetson device.
Everything else (embedding, KNN, gating, research, dashboard) runs on the laptop.

Usage:
    python jetson_client.py --backend http://<laptop-ip>:8000
"""
import argparse
import base64
import time
import cv2
import requests
from ultralytics import YOLO


def encode_crop(crop, max_size=192):
    """Encode crop as small base64 JPEG."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return ""
    scale = min(max_size / max(h, w), 1.0)
    resized = cv2.resize(crop, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode("ascii")


def run_client(backend_url: str, camera_index: int = 0, confidence: float = 0.3,
               target_fps: float = 3.0):
    """Main capture + detect + send loop."""

    print(f"Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    print(f"YOLO loaded. Opening camera {camera_index}...")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        return

    print(f"Camera open. Sending detections to {backend_url}")
    print(f"Target FPS: {target_fps}, min confidence: {confidence}")

    ingest_url = f"{backend_url}/ingest"
    frame_interval = 1.0 / target_fps
    last_frame_time = 0
    frames_sent = 0
    errors = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            now = time.time()
            if now - last_frame_time < frame_interval:
                continue
            last_frame_time = now

            # Run YOLO — use low confidence to catch uncertain objects too
            results = model(frame, conf=confidence, verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = result.names[cls_id]

                    # Crop the object
                    x1, y1, x2, y2 = xyxy
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    crop = frame[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    detections.append({
                        "bbox": xyxy,
                        "yolo_class": cls_name,
                        "yolo_confidence": round(conf, 4),
                        "crop_b64": encode_crop(crop),
                    })

            if not detections:
                continue

            # POST to backend
            payload = {
                "timestamp": now,
                "frame_id": frames_sent,
                "detections": detections,
            }

            try:
                resp = requests.post(ingest_url, json=payload, timeout=2.0)
                if resp.status_code == 200:
                    frames_sent += 1
                    data = resp.json()
                    # Print summary
                    for obj in data.get("objects", []):
                        state = obj.get("state", "?")
                        label = obj.get("label", obj.get("yolo_class", "?"))
                        sim = obj.get("similarity", 0)
                        print(f"  [{state.upper():>9}] {label} "
                              f"(yolo: {obj.get('yolo_class')}@{obj.get('yolo_confidence', 0):.0%}, "
                              f"sim: {sim:.0%})")
                else:
                    errors += 1
                    if errors % 10 == 1:
                        print(f"Backend error {resp.status_code}: {resp.text[:200]}")
            except requests.exceptions.RequestException as e:
                errors += 1
                if errors % 10 == 1:
                    print(f"Connection error ({errors} total): {e}")
                time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\nStopping. Sent {frames_sent} frames, {errors} errors.")
    finally:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenClawdIRL Jetson Client")
    parser.add_argument("--backend", default="http://localhost:8000",
                        help="Backend URL (default: http://localhost:8000)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Min YOLO confidence (default: 0.3, low to catch unknowns)")
    parser.add_argument("--fps", type=float, default=3.0,
                        help="Target frames per second to send (default: 3)")
    args = parser.parse_args()

    run_client(args.backend, args.camera, args.confidence, args.fps)
