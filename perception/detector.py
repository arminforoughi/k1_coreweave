"""Object detection using YOLO."""
import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.5):
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run detection on a frame.

        Returns list of dicts with keys: bbox, confidence, class_id, class_name
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
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

                detections.append({
                    "bbox": xyxy,  # [x1, y1, x2, y2]
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                })

        return detections
