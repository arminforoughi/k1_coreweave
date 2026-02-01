"""Event schemas for Redis Streams communication."""
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ObjectEvent:
    """Emitted by perception for each tracked object."""
    track_id: str
    bbox: list  # [x1, y1, x2, y2]
    state: str  # "known", "uncertain", "unknown"
    label: Optional[str] = None
    similarity: float = 0.0
    thumbnail_b64: str = ""
    embedding: Optional[list] = None
    depth_value: Optional[float] = None
    crop_quality: Optional[float] = None
    yolo_class: str = ""
    yolo_confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_redis(self) -> dict:
        d = asdict(self)
        d["bbox"] = json.dumps(d["bbox"])
        if d["embedding"]:
            d["embedding"] = json.dumps(d["embedding"])
        else:
            d["embedding"] = ""
        return {k: str(v) if v is not None else "" for k, v in d.items()}

    @classmethod
    def from_redis(cls, data: dict) -> "ObjectEvent":
        data["bbox"] = json.loads(data.get("bbox", "[]"))
        emb = data.get("embedding", "")
        data["embedding"] = json.loads(emb) if emb else None
        data["similarity"] = float(data.get("similarity", 0))
        data["yolo_confidence"] = float(data.get("yolo_confidence", 0))
        data["timestamp"] = float(data.get("timestamp", 0))
        dv = data.get("depth_value", "")
        data["depth_value"] = float(dv) if dv else None
        cq = data.get("crop_quality", "")
        data["crop_quality"] = float(cq) if cq else None
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class UnknownEvent:
    """Emitted when an object is classified as unknown."""
    track_id: str
    thumbnail_b64: str = ""
    embedding: Optional[list] = None
    top_similarity: float = 0.0
    depth_value: Optional[float] = None
    crop_quality: Optional[float] = None
    yolo_class: str = ""
    yolo_confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_redis(self) -> dict:
        d = asdict(self)
        if d["embedding"]:
            d["embedding"] = json.dumps(d["embedding"])
        else:
            d["embedding"] = ""
        return {k: str(v) if v is not None else "" for k, v in d.items()}

    @classmethod
    def from_redis(cls, data: dict) -> "UnknownEvent":
        emb = data.get("embedding", "")
        data["embedding"] = json.loads(emb) if emb else None
        data["top_similarity"] = float(data.get("top_similarity", 0))
        data["yolo_confidence"] = float(data.get("yolo_confidence", 0))
        data["timestamp"] = float(data.get("timestamp", 0))
        dv = data.get("depth_value", "")
        data["depth_value"] = float(dv) if dv else None
        cq = data.get("crop_quality", "")
        data["crop_quality"] = float(cq) if cq else None
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LabelEvent:
    """Emitted when a user labels an object."""
    track_id: str
    label_name: str
    label_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_redis(self) -> dict:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_redis(cls, data: dict) -> "LabelEvent":
        data["timestamp"] = float(data.get("timestamp", 0))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
