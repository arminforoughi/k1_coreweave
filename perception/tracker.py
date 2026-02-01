"""Simple IoU-based tracker to assign stable track IDs."""
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Track:
    track_id: str
    bbox: list
    class_name: str
    embedding: Optional[np.ndarray] = None
    frames_seen: int = 1
    last_seen: float = field(default_factory=time.time)
    state: str = "unknown"  # known, uncertain, unknown
    label: Optional[str] = None
    similarity: float = 0.0
    cooldown_until: float = 0.0


def iou(box1: list, box2: list) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


class SimpleTracker:
    """IoU-based tracker that assigns stable track IDs."""

    def __init__(self, iou_threshold: float = 0.3, max_age: float = 2.0):
        self.tracks: dict[str, Track] = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age  # seconds before a track is dropped

    def update(self, detections: list[dict]) -> list[Track]:
        """Match detections to existing tracks, create new tracks as needed."""
        now = time.time()

        # Remove stale tracks
        stale = [tid for tid, t in self.tracks.items() if now - t.last_seen > self.max_age]
        for tid in stale:
            del self.tracks[tid]

        if not detections:
            return list(self.tracks.values())

        # Match detections to tracks via IoU
        matched_tracks = set()
        matched_dets = set()

        existing = list(self.tracks.values())
        for det_idx, det in enumerate(detections):
            best_iou = 0.0
            best_track = None
            for track in existing:
                if track.track_id in matched_tracks:
                    continue
                score = iou(det["bbox"], track.bbox)
                if score > best_iou:
                    best_iou = score
                    best_track = track

            if best_track and best_iou >= self.iou_threshold:
                # Update existing track
                best_track.bbox = det["bbox"]
                best_track.class_name = det["class_name"]
                best_track.frames_seen += 1
                best_track.last_seen = now
                matched_tracks.add(best_track.track_id)
                matched_dets.add(det_idx)

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                track_id = f"track_{self.next_id}"
                self.next_id += 1
                self.tracks[track_id] = Track(
                    track_id=track_id,
                    bbox=det["bbox"],
                    class_name=det["class_name"],
                    last_seen=now,
                )

        return list(self.tracks.values())

    def get_track(self, track_id: str) -> Optional[Track]:
        return self.tracks.get(track_id)
