"""Jetson client: ROS 2 camera + YOLO + MiDaS depth + pre-filter gate + POST.

Runs on the Booster K1 (Jetson Orin NX). Subscribes to the camera via ROS 2,
runs YOLOv8n detection + MiDaS depth estimation, applies a pre-filter gate
(depth + crop quality + confidence), and POSTs qualifying detections to the
backend on the laptop.

Usage:
    python perception/jetson_client.py --backend http://<laptop-ip>:8000

Dependencies (on Jetson):
    - ultralytics, opencv-python-headless, requests, timm, python-dotenv (pip)
    - torch, torchvision (NVIDIA JetPack wheel)
    - rclpy, cv_bridge, sensor_msgs (ROS 2 apt packages)
"""
import argparse
import base64
import os
import sys
import time

# Ensure project root is on sys.path so `from perception.*` and `from shared.*` resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import requests

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

from ultralytics import YOLO

# Depth estimator is optional — gracefully handle if torch.hub fails
try:
    from perception.depth_estimator import DepthEstimator
    HAS_DEPTH = True
except Exception:
    HAS_DEPTH = False


# ---------------------------------------------------------------------------
# Shared helpers (used by both ROS 2 and fallback modes)
# ---------------------------------------------------------------------------

def encode_crop(crop, max_size=192):
    """Encode crop as small base64 JPEG."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return ""
    scale = min(max_size / max(h, w), 1.0)
    resized = cv2.resize(crop, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode("ascii")


def compute_crop_quality(crop):
    """Compute a 0..1 quality score for a crop.

    Combines size (bigger = better) with aspect ratio (square = better).
    A 192x192 crop scores 1.0, a 30x30 scores ~0.024.
    """
    h, w = crop.shape[:2]
    if h < 2 or w < 2:
        return 0.0
    area = h * w
    max_area = 192 * 192
    size_factor = min(area / max_area, 1.0)
    aspect = min(h, w) / max(h, w)
    return round(size_factor * aspect, 4)


def prefilter(depth_value, crop_quality, yolo_confidence, cfg):
    """Pre-filter gate: decide whether to send a detection to the backend.

    Returns:
        "priority" - low confidence but high-quality capture (learning target)
        "send"     - normal detection worth processing
        "skip"     - not worth sending
    """
    # Hard floor: extremely low confidence is noise
    if yolo_confidence < cfg["confidence_floor"]:
        return "skip"

    # Depth gate (if depth is available)
    if depth_value is not None:
        if depth_value < cfg["depth_min"]:
            return "skip"  # too far
        if depth_value > cfg["depth_max"]:
            return "skip"  # too close (camera body / artifact)

    # Crop quality gate
    if crop_quality < cfg["crop_quality_min"]:
        return "skip"

    # Inverted logic: low YOLO confidence + good capture = learning target
    if yolo_confidence < cfg["confidence_low"]:
        close_enough = depth_value is not None and depth_value >= 0.25
        good_crop = crop_quality >= 0.3
        if close_enough and good_crop:
            return "priority"
        else:
            return "skip"

    return "send"


def process_yolo_results(results, frame, depth_map, depth_estimator, prefilter_cfg):
    """Extract detections from YOLO results, apply depth + pre-filter.

    Returns list of detection dicts ready to POST, plus count of filtered.
    """
    detections = []
    filtered = 0

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = result.names[cls_id]

            x1, y1, x2, y2 = xyxy
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Depth at bbox center
            dv = None
            if depth_estimator is not None and depth_map is not None:
                dv = depth_estimator.get_depth_at_bbox(depth_map, [x1, y1, x2, y2])

            # Crop quality
            cq = compute_crop_quality(crop)

            # Pre-filter gate
            action = prefilter(dv, cq, conf, prefilter_cfg)
            if action == "skip":
                filtered += 1
                continue

            detections.append({
                "bbox": xyxy,
                "yolo_class": cls_name,
                "yolo_confidence": round(conf, 4),
                "crop_b64": encode_crop(crop),
                "depth_value": round(dv, 4) if dv is not None else None,
                "crop_quality": cq,
                "prefilter_action": action,
            })

    return detections, filtered


def post_to_backend(ingest_url, payload, timeout=2.0):
    """POST payload to backend, return (success, response_data)."""
    try:
        resp = requests.post(ingest_url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return True, resp.json()
        return False, None
    except requests.exceptions.RequestException:
        return False, None


def log_response(data, logger=None):
    """Log backend response objects."""
    for obj in data.get("objects", []):
        state = obj.get("state", "?")
        label = obj.get("label", obj.get("yolo_class", "?"))
        sim = obj.get("similarity", 0)
        msg = (f"  [{state.upper():>9}] {label} "
               f"(yolo: {obj.get('yolo_class')}@{obj.get('yolo_confidence', 0):.0%}, "
               f"sim: {sim:.0%})")
        if logger:
            logger.info(msg)
        else:
            print(msg)


# ---------------------------------------------------------------------------
# ROS 2 Node (primary mode — used on the Booster K1)
# ---------------------------------------------------------------------------

class JetsonClientNode(Node):
    """ROS 2 node: camera subscription + YOLO + MiDaS + pre-filter + POST."""

    def __init__(self, backend_url, camera_topic, target_fps, yolo_confidence,
                 midas_model, prefilter_cfg):
        super().__init__("openclawdirl_jetson_client")

        self._latest_frame = None
        self._frame_id = 0
        self._frames_sent = 0
        self._frames_filtered = 0
        self._errors = 0
        self._ingest_url = f"{backend_url}/ingest"
        self._prefilter_cfg = prefilter_cfg
        self._yolo_confidence = yolo_confidence

        self.bridge = CvBridge()

        # Load YOLO
        self.get_logger().info("Loading YOLOv8n...")
        self.yolo_model = YOLO("yolov8n.pt")
        self.get_logger().info("YOLO loaded.")

        # Load MiDaS depth (optional)
        self.depth_estimator = None
        if HAS_DEPTH:
            self.get_logger().info(f"Loading MiDaS depth ({midas_model})...")
            try:
                self.depth_estimator = DepthEstimator(model_size=midas_model)
                self.get_logger().info(f"MiDaS loaded on {self.depth_estimator.device}")
            except Exception as e:
                self.get_logger().warn(f"MiDaS failed to load: {e}. Running without depth.")
        else:
            self.get_logger().warn("Depth estimator not available. Running without depth.")

        # ROS 2 camera subscription — match publisher QoS (RELIABLE on K1)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, camera_topic, self._on_image, qos)
        self.get_logger().info(f"Subscribed to {camera_topic}")

        # Timer for processing at target FPS
        interval = 1.0 / target_fps
        self.create_timer(interval, self._process_frame)
        self.get_logger().info(
            f"Ready. Sending to {backend_url} at {target_fps} FPS"
        )

    def _on_image(self, msg):
        """Store latest camera frame. Handles NV12 encoding from Booster camera."""
        try:
            if msg.encoding == "nv12":
                h, w = msg.height, msg.width
                yuv = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (int(h * 1.5), w)
                )
                self._latest_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            else:
                self._latest_frame = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding="bgr8"
                )
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def _process_frame(self):
        """Timer callback: YOLO + MiDaS + pre-filter + POST."""
        frame = self._latest_frame
        if frame is None:
            return
        self._latest_frame = None  # consume frame

        now = time.time()

        # YOLO detection
        results = self.yolo_model(
            frame, conf=self._yolo_confidence, verbose=False
        )

        # MiDaS depth (single pass on full frame)
        depth_map = None
        if self.depth_estimator is not None:
            depth_map = self.depth_estimator.estimate(frame)

        # Process detections through pre-filter
        detections, filtered = process_yolo_results(
            results, frame, depth_map, self.depth_estimator, self._prefilter_cfg
        )
        self._frames_filtered += filtered

        if not detections:
            return

        # POST to backend
        payload = {
            "timestamp": now,
            "frame_id": self._frame_id,
            "detections": detections,
        }
        success, data = post_to_backend(self._ingest_url, payload)
        if success:
            self._frame_id += 1
            self._frames_sent += 1
            log_response(data, logger=self.get_logger())
        else:
            self._errors += 1
            if self._errors % 10 == 1:
                self.get_logger().warn(
                    f"Backend error ({self._errors} total)"
                )


# ---------------------------------------------------------------------------
# Fallback mode (cv2.VideoCapture — for testing without ROS 2)
# ---------------------------------------------------------------------------

def run_fallback(backend_url, camera_index, target_fps, yolo_confidence,
                 midas_model, prefilter_cfg):
    """Fallback loop using cv2.VideoCapture when ROS 2 is not available."""
    print("ROS 2 not available — running in fallback mode (cv2.VideoCapture)")

    print("Loading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    print("YOLO loaded.")

    depth_estimator = None
    if HAS_DEPTH:
        print(f"Loading MiDaS depth ({midas_model})...")
        try:
            depth_estimator = DepthEstimator(model_size=midas_model)
            print(f"MiDaS loaded on {depth_estimator.device}")
        except Exception as e:
            print(f"MiDaS failed to load: {e}. Running without depth.")
    else:
        print("Depth estimator not available. Running without depth.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        return

    ingest_url = f"{backend_url}/ingest"
    frame_interval = 1.0 / target_fps
    last_frame_time = 0
    frame_id = 0
    frames_sent = 0
    frames_filtered = 0
    errors = 0

    print(f"Camera open. Sending to {backend_url} at {target_fps} FPS")

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

            results = model(frame, conf=yolo_confidence, verbose=False)

            depth_map = None
            if depth_estimator is not None:
                depth_map = depth_estimator.estimate(frame)

            detections, filtered = process_yolo_results(
                results, frame, depth_map, depth_estimator, prefilter_cfg
            )
            frames_filtered += filtered

            if not detections:
                continue

            payload = {
                "timestamp": now,
                "frame_id": frame_id,
                "detections": detections,
            }
            success, data = post_to_backend(ingest_url, payload)
            if success:
                frame_id += 1
                frames_sent += 1
                log_response(data)
            else:
                errors += 1
                if errors % 10 == 1:
                    print(f"Backend error ({errors} total)")
                time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\nStopping. Sent {frames_sent}, filtered {frames_filtered}, "
              f"{errors} errors.")
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OpenClawdIRL Jetson Client (ROS 2 + MiDaS depth)"
    )
    parser.add_argument("--backend", default="http://localhost:8000",
                        help="Backend URL (default: http://localhost:8000)")
    parser.add_argument("--topic",
                        default="/booster_camera_bridge/image_left_raw",
                        help="ROS 2 camera topic")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index for fallback mode (default: 0)")
    parser.add_argument("--fps", type=float, default=3.0,
                        help="Target FPS (default: 3)")
    parser.add_argument("--confidence", type=float, default=0.15,
                        help="YOLO confidence floor (default: 0.15)")
    parser.add_argument("--midas-model", default="small",
                        choices=["small", "hybrid", "large"],
                        help="MiDaS model size (default: small)")
    parser.add_argument("--fallback", action="store_true",
                        help="Force cv2.VideoCapture fallback (skip ROS 2)")
    # Pre-filter thresholds (can also be set via env vars)
    parser.add_argument("--depth-min", type=float, default=0.08)
    parser.add_argument("--depth-max", type=float, default=0.95)
    parser.add_argument("--crop-quality-min", type=float, default=0.15)
    parser.add_argument("--confidence-low", type=float, default=0.5)
    args = parser.parse_args()

    prefilter_cfg = {
        "depth_min": args.depth_min,
        "depth_max": args.depth_max,
        "crop_quality_min": args.crop_quality_min,
        "confidence_low": args.confidence_low,
        "confidence_floor": args.confidence,
    }

    if HAS_ROS2 and not args.fallback:
        rclpy.init()
        node = JetsonClientNode(
            backend_url=args.backend,
            camera_topic=args.topic,
            target_fps=args.fps,
            yolo_confidence=args.confidence,
            midas_model=args.midas_model,
            prefilter_cfg=prefilter_cfg,
        )
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info(
                f"Stopping. Sent {node._frames_sent}, "
                f"filtered {node._frames_filtered}, "
                f"{node._errors} errors."
            )
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        run_fallback(
            backend_url=args.backend,
            camera_index=args.camera,
            target_fps=args.fps,
            yolo_confidence=args.confidence,
            midas_model=args.midas_model,
            prefilter_cfg=prefilter_cfg,
        )


if __name__ == "__main__":
    main()
