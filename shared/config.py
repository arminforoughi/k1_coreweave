"""Shared configuration loaded from environment variables."""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RedisConfig:
    url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))


@dataclass
class WeaveConfig:
    api_key: str = field(default_factory=lambda: os.getenv("WANDB_API_KEY", ""))
    project: str = field(default_factory=lambda: os.getenv("WANDB_PROJECT", "openclawdirl"))
    entity: str = field(default_factory=lambda: os.getenv("WANDB_ENTITY", ""))


@dataclass
class PerceptionConfig:
    camera_index: int = field(default_factory=lambda: int(os.getenv("CAMERA_INDEX", "0")))
    detection_confidence: float = field(default_factory=lambda: float(os.getenv("DETECTION_CONFIDENCE", "0.5")))
    unknown_threshold: float = field(default_factory=lambda: float(os.getenv("UNKNOWN_THRESHOLD", "0.4")))
    known_threshold: float = field(default_factory=lambda: float(os.getenv("KNOWN_THRESHOLD", "0.7")))
    margin_threshold: float = field(default_factory=lambda: float(os.getenv("MARGIN_THRESHOLD", "0.15")))
    cooldown_seconds: float = field(default_factory=lambda: float(os.getenv("COOLDOWN_SECONDS", "15")))
    persistence_frames: int = field(default_factory=lambda: int(os.getenv("PERSISTENCE_FRAMES", "10")))
    # Jetson pre-filter gate thresholds
    depth_min: float = field(default_factory=lambda: float(os.getenv("DEPTH_MIN", "0.08")))
    depth_max: float = field(default_factory=lambda: float(os.getenv("DEPTH_MAX", "0.95")))
    crop_quality_min: float = field(default_factory=lambda: float(os.getenv("CROP_QUALITY_MIN", "0.15")))
    confidence_low: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_LOW", "0.5")))
    confidence_floor: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_FLOOR", "0.15")))
    camera_topic: str = field(default_factory=lambda: os.getenv("CAMERA_TOPIC", "/booster_camera_bridge/image_left_raw"))
    midas_model: str = field(default_factory=lambda: os.getenv("MIDAS_MODEL", "small"))


@dataclass
class BackendConfig:
    host: str = field(default_factory=lambda: os.getenv("BACKEND_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("BACKEND_PORT", "8000")))


@dataclass
class Config:
    redis: RedisConfig = field(default_factory=RedisConfig)
    weave: WeaveConfig = field(default_factory=WeaveConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)


def load_config() -> Config:
    return Config()
