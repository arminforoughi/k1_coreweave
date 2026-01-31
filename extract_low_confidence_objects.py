#!/usr/bin/env python3
"""
Extract and save frames containing objects with confidence score below 0.65
Reads from Booster Camera Bridge and uses YOLO for detection
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import argparse
from datetime import datetime
from ultralytics import YOLO


class LowConfidenceExtractor(Node):
    """ROS2 node that extracts and saves low-confidence detections"""

    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.65, output_dir='low_confidence_detections'):
        super().__init__('low_confidence_extractor')

        self.bridge = CvBridge()
        self.confidence_threshold = confidence_threshold
        self.output_dir = output_dir
        self.frame_count = 0
        self.saved_count = 0

        # Create output directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'crops'), exist_ok=True)

        # Initialize YOLO model
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        try:
            self.yolo_model = YOLO(model_path)
            self.get_logger().info('YOLO model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {str(e)}')
            self.yolo_model = None

        # Subscribe to left camera
        left_topic = '/booster_camera_bridge/image_left_raw'
        self.get_logger().info(f'Subscribing to: {left_topic}')
        self.get_logger().info(f'Confidence threshold: < {confidence_threshold}')
        self.get_logger().info(f'Output directory: {self.output_dir}')

        self.subscription = self.create_subscription(
            Image,
            left_topic,
            self.image_callback,
            10
        )

    def detect_low_confidence_objects(self, frame):
        """Detect objects with confidence below threshold"""
        if self.yolo_model is None:
            return []

        try:
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)

            low_conf_detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get detection info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]

                    # Filter for low confidence
                    if confidence < self.confidence_threshold:
                        low_conf_detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name
                        })

            return low_conf_detections
        except Exception as e:
            self.get_logger().error(f'Error in object detection: {str(e)}')
            return []

    def save_frame_with_detections(self, frame, detections):
        """Save frame with bounding boxes for low-confidence objects"""
        if len(detections) == 0:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        # Create annotated frame
        annotated_frame = frame.copy()

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']

            # Draw bounding box in red for low confidence
            color = (0, 0, 255)  # Red for low confidence
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name} {conf:.3f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Save cropped object
            crop = frame[y1:y2, x1:x2]
            crop_filename = f"{timestamp}_obj{idx}_{class_name}_{conf:.3f}.jpg"
            crop_path = os.path.join(self.output_dir, 'crops', crop_filename)
            cv2.imwrite(crop_path, crop)

        # Save full annotated frame
        frame_filename = f"{timestamp}_frame_{len(detections)}objects.jpg"
        frame_path = os.path.join(self.output_dir, 'frames', frame_filename)
        cv2.imwrite(frame_path, annotated_frame)

        self.saved_count += 1

        # Log detection details
        self.get_logger().info(f'Saved frame {self.saved_count}: {len(detections)} low-confidence objects')
        for det in detections:
            self.get_logger().info(f"  - {det['class_name']}: {det['confidence']:.3f}")

    def image_callback(self, msg):
        """Callback for camera image messages"""
        try:
            self.frame_count += 1

            # Convert ROS image to OpenCV format
            if msg.encoding == 'nv12':
                height = msg.height
                width = msg.width
                img_data = np.frombuffer(msg.data, dtype=np.uint8)
                yuv = img_data.reshape((int(height * 1.5), width))
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Detect low confidence objects
            low_conf_detections = self.detect_low_confidence_objects(frame)

            # Save frame if low-confidence objects found
            if len(low_conf_detections) > 0:
                self.save_frame_with_detections(frame, low_conf_detections)

            # Periodic status update
            if self.frame_count % 100 == 0:
                self.get_logger().info(f'Processed {self.frame_count} frames, saved {self.saved_count} with low-confidence detections')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main():
    parser = argparse.ArgumentParser(description='Extract objects with low confidence scores from camera feed')
    parser.add_argument('--threshold', type=float, default=0.65,
                       help='Confidence threshold - save objects below this value (default: 0.65)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--output', type=str, default='low_confidence_detections',
                       help='Output directory for saved frames (default: low_confidence_detections)')

    args = parser.parse_args()

    # Initialize ROS2
    rclpy.init()

    # Create extractor node
    extractor = LowConfidenceExtractor(
        model_path=args.model,
        confidence_threshold=args.threshold,
        output_dir=args.output
    )

    print(f'\n{"="*70}')
    print(f'🔍 Low-Confidence Object Extractor Started')
    print(f'{"="*70}')
    print(f'Topic: /booster_camera_bridge/image_left_raw')
    print(f'Model: {args.model}')
    print(f'Confidence Threshold: < {args.threshold}')
    print(f'Output Directory: {args.output}/')
    print(f'  - Full frames: {args.output}/frames/')
    print(f'  - Object crops: {args.output}/crops/')
    print(f'\nPress Ctrl+C to stop')
    print(f'{"="*70}\n')

    try:
        rclpy.spin(extractor)
    except KeyboardInterrupt:
        print('\n\nShutting down...')
        print(f'Total frames processed: {extractor.frame_count}')
        print(f'Frames saved with low-confidence objects: {extractor.saved_count}')
    finally:
        extractor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
