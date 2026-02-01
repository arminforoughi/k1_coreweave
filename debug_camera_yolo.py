#!/usr/bin/env python3
"""Debug script: Check what YOLO detects from the webcam."""
import cv2
from ultralytics import YOLO

# Load YOLO
print("Loading YOLOv8n...")
model = YOLO("yolov8n.pt")
print("YOLO loaded.\n")

# Open camera
print("Opening camera 0...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera 0")
    exit(1)

print("Camera opened. Press 'q' to quit, 's' to save frame.\n")
print("Detections (confidence >= 0.25):")
print("-" * 60)

frame_count = 0
detection_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame")
            break

        frame_count += 1

        # Run YOLO every 30 frames (~1 second at 30fps)
        if frame_count % 30 == 0:
            results = model(frame, conf=0.25, verbose=False)

            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = model.names[cls]
                        detection_count += 1
                        print(f"Frame {frame_count}: {label} @ {conf:.0%}")

        # Show frame with detections
        results_display = model(frame, conf=0.25, verbose=False)
        annotated = results_display[0].plot()

        cv2.imshow('YOLO Debug', annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"debug_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames, {detection_count} detections")
