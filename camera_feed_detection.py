#!/usr/bin/env python3
"""
Web-based Camera Feed Viewer for Booster Camera Bridge
With Object Detection and Neural Network Stereo Depth Estimation
Uses /booster_camera_bridge topics directly
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import argparse
import io
import os
from datetime import datetime
from ultralytics import YOLO
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image as PILImage

class StereoDepthModel:
    """Neural network-based stereo depth estimation"""

    def __init__(self, model_type='crestereo'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model = None
        self.processor = None

        print(f"Using device: {self.device}")

        if model_type == 'dpt-beit':
            self.load_dpt_beit()
        elif model_type == 'crestereo':
            self.load_crestereo()
        elif model_type == 'midas':
            self.load_midas_stereo()
        else:
            print(f"Unknown model type: {model_type}, using DPT-BEiT")
            self.load_dpt_beit()

    def load_dpt_beit(self):
        """Load Intel DPT-BEiT-Base-384 model for depth estimation"""
        try:
            model_name = "Intel/dpt-beit-base-384"
            print(f"Loading {model_name}...")

            # Load the processor and model
            self.processor = DPTImageProcessor.from_pretrained(model_name)
            self.model = DPTForDepthEstimation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            print("DPT-BEiT-Base-384 model loaded successfully")
        except Exception as e:
            print(f"Failed to load DPT-BEiT model: {e}")
            self.model = None
            self.processor = None

    def load_midas_stereo(self):
        """Load MiDaS model for depth estimation"""
        try:
            # Use MiDaS small model for faster inference on edge devices
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.model.to(self.device)
            self.model.eval()

            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
            print("MiDaS model loaded successfully")
        except Exception as e:
            print(f"Failed to load MiDaS model: {e}")
            self.model = None

    def load_crestereo(self):
        """Load CREStereo model - placeholder for custom implementation"""
        try:
            # Try to load CREStereo if available
            # For now, fall back to MiDaS
            print("CREStereo not available, falling back to MiDaS")
            self.load_midas_stereo()
        except Exception as e:
            print(f"Failed to load CREStereo: {e}")
            self.load_midas_stereo()

    def estimate_depth_dpt(self, img):
        """Estimate depth using DPT-BEiT model"""
        if self.model is None or self.processor is None:
            return None, None

        try:
            with torch.no_grad():
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(img_rgb)

                # Prepare image for the model
                inputs = self.processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Perform inference
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

                # Interpolate to original size
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                depth_map = prediction.cpu().numpy()

                # Normalize for visualization
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_map = (depth_map * 255).astype(np.uint8)

                # Apply colormap
                depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

                return depth_map, depth_colormap

        except Exception as e:
            print(f"Error in DPT depth estimation: {e}")
            return None, None

    def estimate_depth_stereo(self, left_img, right_img):
        """
        Estimate depth using stereo pair
        For DPT-BEiT and MiDaS: average both views for more stable depth
        For true stereo models: use both images directly
        """
        if self.model is None:
            return None, None

        try:
            # Use DPT-BEiT model if loaded
            if self.model_type == 'dpt-beit' and self.processor is not None:
                with torch.no_grad():
                    # Process left image
                    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
                    left_pil = PILImage.fromarray(left_rgb)
                    inputs_left = self.processor(images=left_pil, return_tensors="pt")
                    inputs_left = {k: v.to(self.device) for k, v in inputs_left.items()}
                    outputs_left = self.model(**inputs_left)
                    depth_left = outputs_left.predicted_depth

                    # Process right image
                    right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
                    right_pil = PILImage.fromarray(right_rgb)
                    inputs_right = self.processor(images=right_pil, return_tensors="pt")
                    inputs_right = {k: v.to(self.device) for k, v in inputs_right.items()}
                    outputs_right = self.model(**inputs_right)
                    depth_right = outputs_right.predicted_depth

                    # Average both depth maps
                    depth_avg = (depth_left + depth_right) / 2.0

                    # Interpolate to original size
                    prediction = torch.nn.functional.interpolate(
                        depth_avg,
                        size=left_img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                    depth_map = prediction.cpu().numpy()

                    # Normalize
                    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                    depth_map = (depth_map * 255).astype(np.uint8)

                    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

                    return depth_map, depth_colormap

            # Use MiDaS model
            else:
                with torch.no_grad():
                    # Convert BGR to RGB
                    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
                    right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

                    # Process left image
                    input_batch_left = self.transform(left_rgb).to(self.device)
                    depth_left = self.model(input_batch_left)
                    depth_left = torch.nn.functional.interpolate(
                        depth_left.unsqueeze(1),
                        size=left_img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                    # Process right image
                    input_batch_right = self.transform(right_rgb).to(self.device)
                    depth_right = self.model(input_batch_right)
                    depth_right = torch.nn.functional.interpolate(
                        depth_right.unsqueeze(1),
                        size=right_img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                    # Average both depth maps for more stable result
                    depth_avg = (depth_left + depth_right) / 2.0
                    depth_map = depth_avg.cpu().numpy()

                    # Normalize for visualization
                    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                    depth_map = (depth_map * 255).astype(np.uint8)

                    # Apply colormap
                    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

                    return depth_map, depth_colormap

        except Exception as e:
            print(f"Error in depth estimation: {e}")
            return None, None

    def estimate_depth_mono(self, img):
        """Estimate depth from single image"""
        if self.model is None:
            return None, None

        try:
            # Use DPT-BEiT if loaded
            if self.model_type == 'dpt-beit' and self.processor is not None:
                return self.estimate_depth_dpt(img)

            # Use MiDaS
            with torch.no_grad():
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                input_batch = self.transform(img_rgb).to(self.device)

                depth = self.model(input_batch)
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                depth_map = depth.cpu().numpy()

                # Normalize
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_map = (depth_map * 255).astype(np.uint8)

                depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

                return depth_map, depth_colormap

        except Exception as e:
            print(f"Error in depth estimation: {e}")
            return None, None


class CameraSubscriber(Node):
    """ROS2 node that subscribes to booster camera bridge topics with AI detection"""

    def __init__(self, show_stereo=False, model_path='yolov8n.pt', depth_model='midas'):
        super().__init__('camera_ai_viewer')

        self.bridge = CvBridge()
        self.latest_left_frame = None
        self.latest_right_frame = None
        self.processed_frame = None
        self.show_stereo = show_stereo
        self.depth_colormap = None

        # Initialize YOLO model for object detection
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        try:
            self.yolo_model = YOLO(model_path)
            self.get_logger().info('YOLO model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {str(e)}')
            self.yolo_model = None

        # Initialize neural network depth model
        self.depth_model = None
        if show_stereo:
            self.get_logger().info(f'Loading depth estimation model: {depth_model}')
            try:
                self.depth_model = StereoDepthModel(model_type=depth_model)
                self.get_logger().info('Depth model loaded successfully')
            except Exception as e:
                self.get_logger().error(f'Failed to load depth model: {str(e)}')

        # Use booster camera bridge topics
        left_topic = '/booster_camera_bridge/image_left_raw'
        right_topic = '/booster_camera_bridge/image_right_raw'

        self.get_logger().info(f'Left camera topic: {left_topic}')
        if show_stereo:
            self.get_logger().info(f'Right camera topic: {right_topic}')

        # Subscribe to left camera
        self.left_subscription = self.create_subscription(
            Image,
            left_topic,
            self.left_callback,
            10
        )

        # Subscribe to right camera if requested
        if show_stereo:
            self.right_subscription = self.create_subscription(
                Image,
                right_topic,
                self.right_callback,
                10
            )

    def detect_objects(self, frame):
        """Perform object detection on frame"""
        if self.yolo_model is None:
            return []

        try:
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]

                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    })

            return detections
        except Exception as e:
            self.get_logger().error(f'Error in object detection: {str(e)}')
            return []

    def get_depth_at_bbox(self, depth_map, bbox):
        """Get average depth in bounding box region"""
        if depth_map is None:
            return None

        try:
            x1, y1, x2, y2 = bbox
            # Get central region of bbox for more stable depth reading
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            region_size = min((x2 - x1) // 4, (y2 - y1) // 4, 20)

            # Extract region
            y_start = max(0, cy - region_size)
            y_end = min(depth_map.shape[0], cy + region_size)
            x_start = max(0, cx - region_size)
            x_end = min(depth_map.shape[1], cx + region_size)

            depth_region = depth_map[y_start:y_end, x_start:x_end]

            # Get average depth value
            avg_depth = np.mean(depth_region)
            return avg_depth
        except Exception as e:
            self.get_logger().error(f'Error getting depth at bbox: {str(e)}')
            return None

    def draw_detections(self, frame, detections, depth_map=None):
        """Draw object detections and depth on frame"""
        output = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']

            # Get depth if available
            depth_str = ""
            if depth_map is not None:
                depth_val = self.get_depth_at_bbox(depth_map, det['bbox'])
                if depth_val is not None:
                    # Depth value is normalized 0-255, convert to relative scale
                    depth_str = f" | Depth:{depth_val:.0f}"

            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{class_name} {conf:.2f}{depth_str}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

            # Draw label text
            cv2.putText(output, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return output

    def process_frames(self):
        """Process frames with object detection and neural network depth estimation"""
        if self.latest_left_frame is None:
            return

        left_frame = self.latest_left_frame.copy()

        # Detect objects
        detections = self.detect_objects(left_frame)

        # Compute depth using neural network if stereo is enabled
        depth_map = None
        depth_colormap = None
        if self.show_stereo and self.depth_model is not None:
            if self.latest_right_frame is not None:
                # Use stereo depth estimation with both cameras
                depth_map, depth_colormap = self.depth_model.estimate_depth_stereo(
                    left_frame, self.latest_right_frame
                )
            else:
                # Fall back to monocular depth if right frame not available
                depth_map, depth_colormap = self.depth_model.estimate_depth_mono(left_frame)

        # Draw detections with depth
        self.processed_frame = self.draw_detections(left_frame, detections, depth_map)

        # Add info overlay
        depth_status = "Stereo" if self.latest_right_frame is not None else "Mono"
        info_text = f"Objects: {len(detections)} | Depth: {depth_status}"
        cv2.putText(self.processed_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Store depth colormap for visualization
        self.depth_colormap = depth_colormap

    def left_callback(self, msg):
        """Callback for left camera image messages"""
        try:
            # Handle NV12 encoding
            if msg.encoding == 'nv12':
                height = msg.height
                width = msg.width
                img_data = np.frombuffer(msg.data, dtype=np.uint8)
                yuv = img_data.reshape((int(height * 1.5), width))
                cv_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                self.latest_left_frame = cv_image
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_left_frame = cv_image

            # Process frames after receiving left frame
            self.process_frames()
        except Exception as e:
            self.get_logger().error(f'Error converting left image: {str(e)}')

    def right_callback(self, msg):
        """Callback for right camera image messages"""
        try:
            # Handle NV12 encoding
            if msg.encoding == 'nv12':
                height = msg.height
                width = msg.width
                img_data = np.frombuffer(msg.data, dtype=np.uint8)
                yuv = img_data.reshape((int(height * 1.5), width))
                cv_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                self.latest_right_frame = cv_image
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_right_frame = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting right image: {str(e)}')


class CameraHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for serving camera images with AI processing"""

    camera_node = None

    def log_message(self, format, *args):
        """Override to suppress HTTP request logs"""
        pass

    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            # Serve HTML page
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Booster Camera AI Detection</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #1e1e1e;
                        color: #ffffff;
                        margin: 0;
                        padding: 20px;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }
                    h1 {
                        margin-bottom: 10px;
                    }
                    .info {
                        color: #888;
                        margin-bottom: 20px;
                    }
                    .feed-container {
                        display: flex;
                        gap: 20px;
                        flex-wrap: wrap;
                        justify-content: center;
                    }
                    .feed-box {
                        border: 2px solid #444;
                        padding: 10px;
                        border-radius: 8px;
                        background-color: #2d2d2d;
                    }
                    .feed-box h2 {
                        margin-top: 0;
                        margin-bottom: 10px;
                        font-size: 18px;
                        color: #4CAF50;
                    }
                    img {
                        max-width: 640px;
                        height: auto;
                        display: block;
                        border-radius: 4px;
                    }
                    .badge {
                        background-color: #4CAF50;
                        color: white;
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 12px;
                        margin-left: 8px;
                    }
                </style>
            </head>
            <body>
                <h1>🤖 Booster Camera AI Detection</h1>
                <div class="info">Object Detection + Depth Estimation</div>
                <div class="feed-container">
                    <div class="feed-box">
                        <h2>Detected Objects <span class="badge">YOLO</span></h2>
                        <img id="processed-feed" src="/processed" alt="Processed Feed">
                    </div>
                    """ + ("""
                    <div class="feed-box">
                        <h2>Depth Map <span class="badge">Stereo</span></h2>
                        <img id="depth-feed" src="/depth" alt="Depth Map">
                    </div>
                    """ if self.camera_node and self.camera_node.show_stereo else "") + """
                    <div class="feed-box">
                        <h2>Left Camera (Raw)</h2>
                        <img id="left-feed" src="/left" alt="Left Camera">
                    </div>
                    """ + ("""
                    <div class="feed-box">
                        <h2>Right Camera (Raw)</h2>
                        <img id="right-feed" src="/right" alt="Right Camera">
                    </div>
                    """ if self.camera_node and self.camera_node.show_stereo else "") + """
                </div>
                <script>
                    function refreshImage(id, src) {
                        const img = document.getElementById(id);
                        if (img) {
                            const newSrc = src + '?t=' + new Date().getTime();
                            img.src = newSrc;
                        }
                    }

                    setInterval(() => {
                        refreshImage('processed-feed', '/processed');
                        refreshImage('left-feed', '/left');
                        """ + ("""
                        refreshImage('depth-feed', '/depth');
                        refreshImage('right-feed', '/right');
                        """ if self.camera_node and self.camera_node.show_stereo else "") + """
                    }, 100);
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())

        elif self.path.startswith('/processed'):
            # Serve processed frame with detections
            if self.camera_node and self.camera_node.processed_frame is not None:
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                self.end_headers()

                _, buffer = cv2.imencode('.jpg', self.camera_node.processed_frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.wfile.write(buffer.tobytes())
            else:
                self.send_error(503, 'No processed feed available')

        elif self.path.startswith('/depth'):
            # Serve depth map
            if (self.camera_node and hasattr(self.camera_node, 'depth_colormap') and
                self.camera_node.depth_colormap is not None):
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                self.end_headers()

                _, buffer = cv2.imencode('.jpg', self.camera_node.depth_colormap,
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.wfile.write(buffer.tobytes())
            else:
                self.send_error(503, 'No depth map available')

        elif self.path.startswith('/left'):
            # Serve left camera image
            if self.camera_node and self.camera_node.latest_left_frame is not None:
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                self.end_headers()

                _, buffer = cv2.imencode('.jpg', self.camera_node.latest_left_frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.wfile.write(buffer.tobytes())
            else:
                self.send_error(503, 'No left camera feed available')

        elif self.path.startswith('/right'):
            # Serve right camera image
            if self.camera_node and self.camera_node.latest_right_frame is not None:
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                self.end_headers()

                _, buffer = cv2.imencode('.jpg', self.camera_node.latest_right_frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.wfile.write(buffer.tobytes())
            else:
                self.send_error(503, 'No right camera feed available')
        else:
            self.send_error(404, 'Not found')


def spin_ros(node):
    """Spin ROS2 node in a separate thread"""
    rclpy.spin(node)


def main():
    parser = argparse.ArgumentParser(description='AI-Enhanced Camera Feed Viewer for Booster Camera Bridge')
    parser.add_argument('--stereo', action='store_true',
                       help='Enable stereo depth estimation with both cameras')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind web server to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080,
                       help='Port for web server (default: 8080)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--depth-model', type=str, default='dpt-beit',
                       help='Depth model to use: dpt-beit, midas, or crestereo (default: dpt-beit)')

    args = parser.parse_args()

    # Initialize ROS2
    rclpy.init()

    # Create camera subscriber node with AI
    camera_node = CameraSubscriber(
        show_stereo=args.stereo,
        model_path=args.model,
        depth_model=args.depth_model
    )

    # Set the camera node for the HTTP handler
    CameraHTTPHandler.camera_node = camera_node

    # Start ROS2 spinning in a separate thread
    ros_thread = threading.Thread(target=spin_ros, args=(camera_node,), daemon=True)
    ros_thread.start()

    # Start HTTP server
    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, CameraHTTPHandler)

    print(f'\n{"="*60}')
    print(f'🤖 Booster Camera AI Detection Viewer Started')
    print(f'{"="*60}')
    print(f'Using topics: /booster_camera_bridge/image_left_raw')
    if args.stereo:
        print(f'             /booster_camera_bridge/image_right_raw')
    print(f'Object Detection: YOLOv8 ({args.model})')
    if args.stereo:
        print(f'Depth Model: Neural Network ({args.depth_model.upper()})')
        print(f'Depth Mode: Stereo (using both cameras)')
    else:
        print(f'Depth Estimation: Disabled (use --stereo to enable)')
    print(f'Web server: http://{args.host}:{args.port}')
    print(f'\nOpen this URL in your browser to view the AI-enhanced feed')
    print(f'Press Ctrl+C to stop')
    print(f'{"="*60}\n')

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n\nShutting down...')
        httpd.shutdown()
        camera_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
