#!/usr/bin/env python3
"""
K1 Robot - Person Following with Object Detection and Depth Estimation
Combines YOLO detection, DPT depth estimation, and B1 locomotion control
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from datetime import datetime
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
import sys
import time

# Import Booster SDK
from booster_robotics_sdk_python import B1LocoClient, ChannelFactory, RobotMode

class PersonFollowerNode(Node):
    """ROS2 node for person detection, depth estimation, and following"""

    # MiDaS model configurations
    DEPTH_MODELS = {
        'small': 'MiDaS_small',      # Fastest, ~10 FPS
        'hybrid': 'DPT_Hybrid',      # Balanced
        'large': 'DPT_Large'         # Most accurate
    }

    def __init__(self, loco_client, yolo_model='yolov8n.pt', depth_model='small',
                 show_depth=True, confidence=0.5, use_web=False):
        super().__init__('person_follower_node')

        self.bridge = CvBridge()
        self.loco_client = loco_client
        self.latest_left_frame = None
        self.show_depth = show_depth
        self.use_web = use_web
        self.confidence = confidence

        # Following control
        self.follow_enabled = False
        self.lock = threading.Lock()

        # Detection and depth results
        self.latest_detections = []
        self.latest_depth_map = None
        self.latest_depth_colormap = None
        self.latest_annotated_frame = None
        self.target_person = None

        # Person selection and tracking
        self.locked_person_id = None  # ID of locked person
        self.person_last_position = None  # Track position for persistence

        # Following parameters (MiDaS depth: 0=far, 1=close)
        # Tuned for 1-2 meter following distance
        self.target_depth_min = 0.2   # Move forward if farther (increased range)
        self.target_depth_max = 0.82  # Back up ONLY if closer than ~1m
        self.center_tolerance = 0.15  # How centered person needs to be (fraction of width)

        # Movement speeds
        self.forward_speed = 0.3
        self.rotation_speed = 0.4
        self.backward_speed = 0.15

        # Obstacle avoidance
        self.obstacle_threshold = 0.75  # Stop if obstacles closer than this
        self.obstacle_check_width = 0.3  # Check center 30% of frame width

        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # Load YOLO model
        self.get_logger().info(f'Loading YOLO model: {yolo_model}')
        try:
            self.yolo_model = YOLO(yolo_model)
            self.get_logger().info('YOLO model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {str(e)}')
            raise

        # Load MiDaS depth estimation model
        depth_model_name = self.DEPTH_MODELS.get(depth_model, self.DEPTH_MODELS['small'])
        self.get_logger().info(f'Loading MiDaS depth model: {depth_model_name}')
        try:
            self.depth_model = torch.hub.load("intel-isl/MiDaS", depth_model_name, trust_repo=True)
            self.depth_model.to(self.device)
            self.depth_model.eval()

            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if depth_model_name == 'DPT_Large' or depth_model_name == 'DPT_Hybrid':
                self.depth_transform = midas_transforms.dpt_transform
            else:
                self.depth_transform = midas_transforms.small_transform

            self.get_logger().info('MiDaS depth model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load MiDaS model: {str(e)}')
            raise

        # Camera topics
        left_topic = '/booster_camera_bridge/image_left_raw'
        self.get_logger().info(f'Subscribing to: {left_topic}')

        # Subscribe to left camera
        self.left_subscription = self.create_subscription(
            Image, left_topic, self.left_callback, 10
        )

        # Processing timer
        self.timer = self.create_timer(0.15, self.process_frames)

        self.get_logger().info('Person Follower node initialized')

    def convert_image(self, msg):
        """Convert ROS image message to OpenCV format"""
        try:
            if msg.encoding == 'nv12':
                height = msg.height
                width = msg.width
                img_data = np.frombuffer(msg.data, dtype=np.uint8)
                yuv = img_data.reshape((int(height * 1.5), width))
                cv_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                return cv_image
            else:
                return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {str(e)}')
            return None

    def left_callback(self, msg):
        """Callback for left camera"""
        cv_image = self.convert_image(msg)
        if cv_image is not None:
            self.latest_left_frame = cv_image

    def estimate_depth_midas(self, image):
        """Estimate depth using MiDaS model"""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply transforms
            input_batch = self.depth_transform(image_rgb).to(self.device)

            # Run inference
            with torch.no_grad():
                prediction = self.depth_model(input_batch)

                # Resize to original resolution
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Convert to numpy
            depth = prediction.cpu().numpy()

            # Normalize for visualization
            # MiDaS outputs inverse depth: large values = close, small values = far
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
            # Keep as-is: large values (close to 1) = close, small values = far
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)

            # Apply colormap
            depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

            return depth_normalized, depth_colormap

        except Exception as e:
            self.get_logger().error(f'Error estimating depth: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None, None

    def get_depth_at_point(self, depth_map, x, y, window_size=10):
        """Get depth value at a specific point"""
        if depth_map is None:
            return None

        h, w = depth_map.shape
        x, y = int(x), int(y)

        if x < 0 or x >= w or y < 0 or y >= h:
            return None

        half_win = window_size // 2
        y1 = max(0, y - half_win)
        y2 = min(h, y + half_win + 1)
        x1 = max(0, x - half_win)
        x2 = min(w, x + half_win + 1)

        window = depth_map[y1:y2, x1:x2]
        return float(np.mean(window))

    def check_obstacles_ahead(self, depth_map):
        """Check for obstacles in front of robot using depth map"""
        if depth_map is None:
            return False

        h, w = depth_map.shape

        # Check center region in front (bottom-center of image)
        center_x = w // 2
        width_check = int(w * self.obstacle_check_width)
        x1 = center_x - width_check // 2
        x2 = center_x + width_check // 2

        # Check bottom half of image (where floor/obstacles are)
        y1 = h // 2
        y2 = h

        # Get depth values in this region
        region = depth_map[y1:y2, x1:x2]

        # Check if any close obstacles (high depth values = close)
        max_depth = np.max(region)

        return max_depth > self.obstacle_threshold

    def find_target_person(self, detections, frame_width):
        """Find the best person to follow"""
        persons = [d for d in detections if d['class'] == 'person']

        if not persons:
            return None

        # Assign IDs to persons based on position
        for i, person in enumerate(persons):
            person['id'] = i

        # If locked onto a specific person, try to find them
        if self.locked_person_id is not None and self.locked_person_id < len(persons):
            # Use position-based tracking to maintain lock
            if self.person_last_position is not None:
                # Find person closest to last known position
                last_x, last_y = self.person_last_position
                closest_person = min(persons, key=lambda p:
                    ((p['center'][0] - last_x)**2 + (p['center'][1] - last_y)**2)**0.5)

                # If close enough to last position, keep tracking
                distance = ((closest_person['center'][0] - last_x)**2 +
                           (closest_person['center'][1] - last_y)**2)**0.5
                if distance < 200:  # Within 200 pixels
                    self.person_last_position = closest_person['center']
                    return closest_person

            # Fallback: use locked ID if still valid
            if self.locked_person_id < len(persons):
                self.person_last_position = persons[self.locked_person_id]['center']
                return persons[self.locked_person_id]

        # No lock or lock lost - find person closest to center
        center_x = frame_width / 2
        best_person = min(persons, key=lambda p: abs(p['center'][0] - center_x))
        self.person_last_position = best_person['center']

        return best_person

    def select_person(self, person_id):
        """Lock onto a specific person by ID"""
        with self.lock:
            self.locked_person_id = person_id
            self.person_last_position = None
            self.get_logger().info(f"Locked onto person #{person_id}")

    def calculate_movement(self, person, frame_width, frame_height):
        """Calculate movement commands based on person position and depth"""
        if person is None:
            return 0.0, 0.0, 0.0  # stop

        # Get person's center position
        center_x, center_y = person['center']
        depth_value = person['depth']

        if depth_value is None:
            return 0.0, 0.0, 0.0  # stop if no depth info

        # Calculate horizontal offset from center (normalized -1 to 1)
        frame_center = frame_width / 2
        offset = (center_x - frame_center) / frame_center

        # Initialize movement commands
        x_vel = 0.0  # forward/backward
        y_vel = 0.0  # strafe (not used)
        z_vel = 0.0  # rotation

        # Rotation control (turn to face person)
        if abs(offset) > self.center_tolerance:
            # Turn towards person (negative offset = person on left, rotate left)
            z_vel = -offset * self.rotation_speed
            z_vel = np.clip(z_vel, -0.5, 0.5)

        # Forward/backward control based on depth
        # Note: MiDaS depth - higher values = CLOSER, lower values = FARTHER
        if depth_value > self.target_depth_max:
            # Too close (< 1m), back up
            x_vel = -self.backward_speed
        elif depth_value < self.target_depth_min:
            # Too far (> 2m), move forward
            x_vel = self.forward_speed
        else:
            # In good range (1-2m), just track rotation
            x_vel = 0.0

        # Only move forward if reasonably centered
        if abs(offset) > 0.3 and x_vel > 0:
            x_vel *= 0.5  # Slow down if not centered

        return x_vel, y_vel, z_vel

    def process_frames(self):
        """Process frames for detection, depth, and following"""
        if self.latest_left_frame is None:
            return

        try:
            # Run YOLO detection
            results = self.yolo_model(self.latest_left_frame, conf=self.confidence, verbose=False)

            # Compute MiDaS depth
            depth_map, depth_colormap = self.estimate_depth_midas(self.latest_left_frame)
            self.latest_depth_map = depth_map
            self.latest_depth_colormap = depth_colormap

            # Get detections
            detections = []
            annotated_frame = self.latest_left_frame.copy()
            frame_height, frame_width = annotated_frame.shape[:2]

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.yolo_model.names[cls]

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    depth_value = self.get_depth_at_point(depth_map, center_x, center_y)

                    detection = {
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (int(center_x), int(center_y)),
                        'depth': depth_value
                    }
                    detections.append(detection)

                    # Draw bounding box
                    color = (0, 255, 0) if class_name == 'person' else (255, 100, 0)
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # Label with person ID if it's a person
                    if class_name == 'person':
                        person_id = len([d for d in detections if d['class'] == 'person' and
                                       d['bbox'][1] < y1])  # Count persons above this one
                        if depth_value is not None:
                            label = f'Person #{person_id} | D:{depth_value:.2f}'
                        else:
                            label = f'Person #{person_id}'
                    else:
                        if depth_value is not None:
                            label = f'{class_name} {conf:.2f} | D:{depth_value:.2f}'
                        else:
                            label = f'{class_name} {conf:.2f}'

                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (int(x1), int(y1) - 25),
                                (int(x1) + label_w + 10, int(y1)), color, -1)
                    cv2.putText(annotated_frame, label, (int(x1) + 5, int(y1) - 8),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.circle(annotated_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

            self.latest_detections = detections

            # Find target person (always, even if follow is off)
            target_person = self.find_target_person(detections, frame_width)
            self.target_person = target_person

            # Always track person with head rotation
            if target_person:
                # Calculate head rotation based on person position
                center_x = target_person['center'][0]
                frame_center = frame_width / 2
                offset = (center_x - frame_center) / frame_center  # -1 to 1

                # Add dead zone to prevent jitter
                dead_zone = 0.1
                if abs(offset) < dead_zone:
                    offset = 0.0

                # Convert to yaw angle (radians)
                # Person on left (offset < 0) → turn left (positive yaw)
                # Person on right (offset > 0) → turn right (negative yaw)
                max_yaw = 0.6  # ~34 degrees max (reduced from 45)
                yaw = -offset * max_yaw  # Inverted sign
                pitch = 0.0  # Keep head level

                # Send head rotation command
                self.loco_client.RotateHead(pitch, yaw)
            else:
                # No person detected, center head
                self.loco_client.RotateHead(0.0, 0.0)

            # Movement control (only when follow mode enabled)
            with self.lock:
                if self.follow_enabled:
                    # Draw target indicator
                    if target_person:
                        x1, y1, x2, y2 = target_person['bbox']
                        cv2.rectangle(annotated_frame, (x1-5, y1-5), (x2+5, y2+5), (0, 255, 255), 3)
                        lock_status = "LOCKED" if self.locked_person_id is not None else "AUTO"
                        cv2.putText(annotated_frame, f"TARGET ({lock_status})", (x1, y1 - 35),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # Calculate movement commands
                    x_vel, y_vel, z_vel = self.calculate_movement(target_person, frame_width, frame_height)

                    # Check for obstacles ahead (only if moving forward)
                    obstacle_detected = False
                    if x_vel > 0:  # Only check when moving forward
                        obstacle_detected = self.check_obstacles_ahead(depth_map)
                        if obstacle_detected:
                            x_vel = 0.0  # Stop forward movement
                            self.get_logger().warn("Obstacle detected! Stopping forward movement")

                    # Send movement command
                    self.loco_client.Move(x_vel, y_vel, z_vel)

                    # Add obstacle warning to frame
                    if obstacle_detected:
                        cv2.putText(annotated_frame, "OBSTACLE AHEAD!", (10, 80),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    # Log
                    if target_person:
                        self.get_logger().info(
                            f"Following person - Depth:{target_person['depth']:.2f}, "
                            f"Move:({x_vel:.2f}, {y_vel:.2f}, {z_vel:.2f})"
                        )
                    else:
                        self.get_logger().info("No person detected - stopped")
                else:
                    # Follow mode off - just track with head
                    if target_person:
                        self.get_logger().info(f"Tracking person with head - Depth:{target_person['depth']:.2f}")

            # Add status overlay
            status_text = "FOLLOW MODE: ON" if self.follow_enabled else "FOLLOW MODE: OFF"
            status_color = (0, 255, 0) if self.follow_enabled else (0, 0, 255)
            cv2.putText(annotated_frame, status_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

            self.latest_annotated_frame = annotated_frame

        except Exception as e:
            self.get_logger().error(f'Error processing frames: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def enable_follow(self):
        """Enable person following mode"""
        with self.lock:
            self.follow_enabled = True
            self.get_logger().info("Person following ENABLED")

    def disable_follow(self):
        """Disable person following mode and stop movement"""
        with self.lock:
            self.follow_enabled = False
            self.loco_client.Move(0.0, 0.0, 0.0)
            self.get_logger().info("Person following DISABLED")

    def toggle_follow(self):
        """Toggle person following mode"""
        with self.lock:
            self.follow_enabled = not self.follow_enabled
            if not self.follow_enabled:
                self.loco_client.Move(0.0, 0.0, 0.0)
            status = "ENABLED" if self.follow_enabled else "DISABLED"
            self.get_logger().info(f"Person following {status}")
            return self.follow_enabled

    def get_display_frame(self):
        """Get composite frame for display"""
        if self.latest_annotated_frame is None:
            return None

        if self.show_depth and self.latest_depth_colormap is not None:
            h, w = self.latest_annotated_frame.shape[:2]
            depth_resized = cv2.resize(self.latest_depth_colormap, (w, h))
            combined = np.hstack([self.latest_annotated_frame, depth_resized])

            cv2.putText(combined, "Object Detection + Following", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "MiDaS Depth Map", (w + 10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return combined
        else:
            return self.latest_annotated_frame


class WebServerHandler(BaseHTTPRequestHandler):
    """HTTP handler for web-based viewing"""
    node = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>K1 Person Follower</title>
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
                    h1 { margin-bottom: 5px; }
                    .subtitle { color: #4CAF50; margin-bottom: 10px; font-size: 14px; }
                    .controls { margin: 20px 0; }
                    button {
                        background-color: #4CAF50;
                        color: white;
                        padding: 15px 30px;
                        font-size: 18px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        margin: 0 10px;
                    }
                    button:hover { background-color: #45a049; }
                    button.stop { background-color: #f44336; }
                    button.stop:hover { background-color: #da190b; }
                    .info { color: #888; margin-bottom: 20px; }
                    img { max-width: 95%; border: 2px solid #444; border-radius: 8px; }
                </style>
            </head>
            <body>
                <h1>K1 Person Following System</h1>
                <div class="subtitle">YOLO Detection + MiDaS Depth + Autonomous Following</div>
                <div class="controls">
                    <button onclick="toggleFollow()">Toggle Follow Mode</button>
                    <button class="stop" onclick="stopFollow()">Emergency Stop</button>
                    <button onclick="unlockPerson()">Auto Select</button>
                </div>
                <div class="info">
                    Click person number to lock tracking | Auto-refreshing feed
                    <div id="person-select" style="margin-top: 10px;"></div>
                </div>
                <img id="feed" src="/frame" alt="Detection Feed">
                <script>
                    function refresh() {
                        const img = document.getElementById('feed');
                        img.src = '/frame?t=' + new Date().getTime();
                    }
                    setInterval(refresh, 150);

                    function toggleFollow() {
                        fetch('/toggle').then(r => r.text()).then(msg => console.log(msg));
                    }

                    function stopFollow() {
                        fetch('/stop').then(r => r.text()).then(msg => console.log(msg));
                    }

                    function selectPerson(id) {
                        fetch('/select/' + id).then(r => r.text()).then(msg => {
                            console.log(msg);
                            alert('Locked onto Person #' + id);
                        });
                    }

                    function unlockPerson() {
                        fetch('/unlock').then(r => r.text()).then(msg => {
                            console.log(msg);
                            alert('Auto-select mode enabled');
                        });
                    }

                    // Update person selection buttons
                    function updatePersonButtons() {
                        fetch('/persons').then(r => r.json()).then(data => {
                            const div = document.getElementById('person-select');
                            if (data.count > 1) {
                                let html = 'Select person: ';
                                for (let i = 0; i < data.count; i++) {
                                    html += `<button onclick="selectPerson(${i})" style="margin: 0 5px; padding: 5px 10px;">Person #${i}</button>`;
                                }
                                div.innerHTML = html;
                            } else {
                                div.innerHTML = '';
                            }
                        }).catch(e => {});
                    }
                    setInterval(updatePersonButtons, 1000);
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())

        elif self.path.startswith('/frame'):
            if self.node:
                frame = self.node.get_display_frame()
                if frame is not None:
                    try:
                        self.send_response(200)
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Cache-Control', 'no-cache')
                        self.end_headers()
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        self.wfile.write(buffer.tobytes())
                    except (BrokenPipeError, ConnectionResetError):
                        # Client disconnected, ignore gracefully
                        pass
                else:
                    self.send_error(503, 'No frame available')
            else:
                self.send_error(503, 'Node not ready')

        elif self.path == '/toggle':
            if self.node:
                status = self.node.toggle_follow()
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                msg = f"Follow mode: {'ENABLED' if status else 'DISABLED'}"
                self.wfile.write(msg.encode())
            else:
                self.send_error(503, 'Node not ready')

        elif self.path == '/stop':
            if self.node:
                self.node.disable_follow()
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Follow mode STOPPED')
            else:
                self.send_error(503, 'Node not ready')

        elif self.path.startswith('/select/'):
            if self.node:
                try:
                    person_id = int(self.path.split('/')[-1])
                    self.node.select_person(person_id)
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    msg = f'Locked onto person #{person_id}'
                    self.wfile.write(msg.encode())
                except ValueError:
                    self.send_error(400, 'Invalid person ID')
            else:
                self.send_error(503, 'Node not ready')

        elif self.path == '/unlock':
            if self.node:
                with self.node.lock:
                    self.node.locked_person_id = None
                    self.node.person_last_position = None
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Auto-select enabled')
            else:
                self.send_error(503, 'Node not ready')

        elif self.path == '/persons':
            if self.node:
                persons = [d for d in self.node.latest_detections if d['class'] == 'person']
                import json
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'count': len(persons)}).encode())
            else:
                self.send_error(503, 'Node not ready')

        else:
            self.send_error(404, 'Not found')


def spin_ros(node):
    """Spin ROS2 in separate thread"""
    rclpy.spin(node)


def main():
    parser = argparse.ArgumentParser(
        description='K1 Robot - Person Following with Detection and Depth',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('network_interface', type=str,
                       help='Network interface for robot connection (e.g., eth0)')
    parser.add_argument('--yolo', type=str, default='yolov8n.pt',
                       help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--depth', type=str, default='small',
                       choices=['small', 'hybrid', 'large'],
                       help='Depth model size: small/hybrid/large (default: small)')
    parser.add_argument('--show-depth', action='store_true',
                       help='Show depth visualization side-by-side')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--web', action='store_true',
                       help='Start web server for viewing')
    parser.add_argument('--port', type=int, default=8080,
                       help='Web server port (default: 8080)')
    parser.add_argument('--auto-follow', action='store_true',
                       help='Enable follow mode automatically on startup')

    args = parser.parse_args()

    # Initialize Booster SDK
    print("Initializing Booster SDK...")
    ChannelFactory.Instance().Init(0, args.network_interface)
    loco_client = B1LocoClient()
    loco_client.Init()
    time.sleep(1.0)

    # Set robot to walking mode
    print("Setting robot to walking mode...")
    res = loco_client.ChangeMode(RobotMode.kWalking)
    if res != 0:
        print(f"Warning: Failed to set walking mode (error {res})")
    else:
        print("Robot is in walking mode")

    # Initialize ROS2
    rclpy.init()

    # Create person follower node
    node = PersonFollowerNode(
        loco_client=loco_client,
        yolo_model=args.yolo,
        depth_model=args.depth,
        show_depth=args.show_depth,
        confidence=args.confidence,
        use_web=args.web
    )

    # Enable follow mode if requested
    if args.auto_follow:
        node.enable_follow()

    # Start ROS2 in thread
    ros_thread = threading.Thread(target=spin_ros, args=(node,), daemon=True)
    ros_thread.start()

    print(f'\n{"="*60}')
    print(f'K1 Person Following System')
    print(f'{"="*60}')
    print(f'YOLO Model: {args.yolo}')
    print(f'Depth Model: {args.depth}')
    print(f'Network: {args.network_interface}')
    print(f'Follow mode: {"ENABLED" if args.auto_follow else "DISABLED"}')

    try:
        if args.web:
            WebServerHandler.node = node
            server = HTTPServer(('0.0.0.0', args.port), WebServerHandler)
            print(f'\nWeb interface: http://0.0.0.0:{args.port}')
            print(f'Controls:')
            print(f'  - Click "Toggle Follow Mode" to start/stop following')
            print(f'  - Click "Emergency Stop" to immediately stop')
            print(f'Press Ctrl+C to exit')
            print(f'{"="*60}\n')
            server.serve_forever()
        else:
            print(f'\nRunning in headless mode')
            print(f'Controls:')
            print(f'  - Press ENTER to toggle follow mode')
            print(f'  - Press Ctrl+C to exit')
            print(f'{"="*60}\n')

            import select
            while rclpy.ok():
                # Check for enter key (non-blocking)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    sys.stdin.readline()
                    node.toggle_follow()
                time.sleep(0.1)

    except KeyboardInterrupt:
        print('\n\nShutting down...')
    finally:
        node.disable_follow()
        node.destroy_node()
        rclpy.shutdown()
        print("Shutdown complete")


if __name__ == '__main__':
    main()
