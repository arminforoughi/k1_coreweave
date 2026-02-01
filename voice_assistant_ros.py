#!/usr/bin/env python3
"""
Voice Assistant with ROS2 Camera Integration
Uses Booster Camera Bridge topics for video feed
"""

import asyncio
import base64
import os
import io
import threading
import cv2
import numpy as np
from typing import Optional
from dotenv import load_dotenv

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Voice assistant imports  
from voice_assistant import (
    VisionFrameProvider, 
    app as voice_app,
    vision_provider as _,
    voice_bot as __,
)
import voice_assistant

# Detection
from ultralytics import YOLO

load_dotenv()


class VoiceAssistantROSNode(Node):
    """ROS2 node that provides camera frames to the voice assistant."""
    
    def __init__(self, vision_provider: VisionFrameProvider, model_path: str = "yolov8n.pt"):
        super().__init__('voice_assistant')
        
        self.vision_provider = vision_provider
        self.bridge = CvBridge()
        self.latest_frame = None
        
        # Load YOLO model
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        try:
            self.yolo_model = YOLO(model_path)
            self.get_logger().info('YOLO model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {e}')
            self.yolo_model = None
        
        # Subscribe to camera topic
        camera_topic = '/booster_camera_bridge/image_left_raw'
        self.get_logger().info(f'Subscribing to: {camera_topic}')
        
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.camera_callback,
            10
        )
        
        # Async loop reference for updating vision provider
        self.loop = None
    
    def detect_objects(self, frame) -> list:
        """Run YOLO detection on frame."""
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(frame, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(box.conf[0]),
                            'class_id': int(box.cls[0]),
                            'class_name': result.names[int(box.cls[0])]
                        })
            
            return detections
        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')
            return []
    
    def camera_callback(self, msg):
        """Process incoming camera frames."""
        try:
            # Handle NV12 encoding (common on Jetson)
            if msg.encoding == 'nv12':
                height = msg.height
                width = msg.width
                img_data = np.frombuffer(msg.data, dtype=np.uint8)
                yuv = img_data.reshape((int(height * 1.5), width))
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            self.latest_frame = frame
            
            # Run detection
            detections = self.detect_objects(frame)
            
            # Draw detections on frame for visualization
            display_frame = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Update vision provider (thread-safe via asyncio)
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self.vision_provider.update_frame(display_frame, detections),
                    self.loop
                )
            
        except Exception as e:
            self.get_logger().error(f'Camera callback error: {e}')


def spin_ros(node):
    """Spin ROS2 node in separate thread."""
    rclpy.spin(node)


async def run_ros_voice_assistant(host: str = "0.0.0.0", port: int = 8090):
    """Run voice assistant with ROS2 camera integration."""
    
    print("🤖 K1 Voice Assistant - ROS2 Mode")
    print("=" * 50)
    
    # Initialize vision provider
    vision_provider = VisionFrameProvider()
    voice_assistant.vision_provider = vision_provider
    voice_assistant.voice_bot = voice_assistant.SimplifiedVoiceBot(vision_provider)
    
    # Initialize ROS2
    rclpy.init()
    
    # Create ROS2 node
    ros_node = VoiceAssistantROSNode(vision_provider)
    ros_node.loop = asyncio.get_event_loop()
    
    # Start ROS2 spinning in background thread
    ros_thread = threading.Thread(target=spin_ros, args=(ros_node,), daemon=True)
    ros_thread.start()
    print("✓ ROS2 node started")
    print("✓ Subscribed to /booster_camera_bridge/image_left_raw")
    
    # Run FastAPI server
    print(f"\n🌐 Voice assistant running at http://{host}:{port}")
    print("📷 Open in browser to use voice interface")
    print("🎤 Ask 'What do you see?' to describe the camera view")
    print("Press Ctrl+C to stop\n")
    
    import uvicorn
    config = uvicorn.Config(voice_app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="K1 Voice Assistant with ROS2")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8090, help="Server port")
    
    args = parser.parse_args()
    
    asyncio.run(run_ros_voice_assistant(args.host, args.port))

