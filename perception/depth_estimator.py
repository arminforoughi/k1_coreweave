"""MiDaS depth estimator for Jetson Orin NX.

Wraps MiDaS Small (default) for monocular depth estimation.
Outputs normalized inverse-depth maps: higher values = closer objects.

Based on depth estimation logic from k1_coreweave/camera_feed_detection.py
and k1_coreweave/k1_follow_person.py.
"""
import cv2
import numpy as np
import torch


class DepthEstimator:
    """Runs MiDaS on CUDA (or CPU fallback), returns normalized depth maps."""

    MODELS = {
        "small": "MiDaS_small",
        "hybrid": "DPT_Hybrid",
        "large": "DPT_Large",
    }

    def __init__(self, model_size: str = "small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = self.MODELS.get(model_size, self.MODELS["small"])

        self.model = torch.hub.load(
            "intel-isl/MiDaS", model_name, trust_repo=True
        )
        self.model.to(self.device)
        self.model.train(False)

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        if model_size in ("hybrid", "large"):
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def estimate(self, bgr_frame: np.ndarray):
        """Return normalized inverse-depth map (float32, 0..1, higher = closer).

        Returns None on failure.
        """
        try:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(rgb).to(self.device)

            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=bgr_frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth = prediction.cpu().numpy()
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min < 1e-6:
                return np.zeros_like(depth, dtype=np.float32)
            return ((depth - d_min) / (d_max - d_min)).astype(np.float32)
        except Exception:
            return None

    def get_depth_at_bbox(self, depth_map: np.ndarray, bbox: list) -> float:
        """Sample center region of a bbox and return mean inverse-depth.

        Replicates the logic from camera_feed_detection.py get_depth_at_bbox:
        uses a small window around the bbox center (max 20px radius) for a
        stable reading.

        Returns None if depth_map is None or bbox is invalid.
        """
        if depth_map is None:
            return None
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        region_size = min((x2 - x1) // 4, (y2 - y1) // 4, 20)

        y_start = max(0, cy - region_size)
        y_end = min(depth_map.shape[0], cy + region_size)
        x_start = max(0, cx - region_size)
        x_end = min(depth_map.shape[1], cx + region_size)

        region = depth_map[y_start:y_end, x_start:x_end]
        if region.size == 0:
            return None
        return float(np.mean(region))
