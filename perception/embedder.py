"""Compute embeddings for object crops using MobileNetV2."""
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image


class Embedder:
    def __init__(self):
        # Use MobileNetV2 as a feature extractor (fast, 1280-dim after pooling)
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Remove the classifier head -- keep features
        self.model.classifier = torch.nn.Identity()
        self.model.train(False)  # set to inference mode

        # Check for CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Compute embedding for an object crop (BGR numpy array).

        Returns a normalized 1280-dim vector.
        """
        # Convert BGR to RGB PIL Image
        rgb = crop[:, :, ::-1] if crop.shape[2] == 3 else crop
        pil_img = Image.fromarray(rgb)

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)

        embedding = features.squeeze().cpu().numpy()
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @property
    def dim(self) -> int:
        return 1280  # MobileNetV2 feature dim
