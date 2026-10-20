"""Single-image inference used by the agent and the API."""
import json
from functools import lru_cache

import torch
from PIL import Image

from histo_agent.vision.dataset import build_transforms
from histo_agent.vision.model import load_checkpoint


class VisionPredictor:
    def __init__(self, ckpt_path: str, threshold_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, ckpt = load_checkpoint(ckpt_path, self.device)
        self.tf = build_transforms(ckpt["img_size"], train=False)
        self.threshold = json.load(open(threshold_path))["threshold"]

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        x = self.tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        p = float(torch.sigmoid(self.model(x)).item())
        return {"p_malignant": p, "threshold": self.threshold,
                "label": "malignant" if p >= self.threshold else "benign"}


@lru_cache(maxsize=1)
def get_predictor(ckpt_path: str, threshold_path: str) -> VisionPredictor:
    return VisionPredictor(ckpt_path, threshold_path)
