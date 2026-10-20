"""Grad-CAM heatmap for one image.

Usage: python -m histo_agent.vision.gradcam --image x.png --out artifacts/gradcam.png
"""
import argparse

import numpy as np
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from histo_agent.vision.dataset import build_transforms
from histo_agent.vision.model import load_checkpoint


def last_conv_layer(model):
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default="artifacts/vision/best.pt")
    ap.add_argument("--out", default="artifacts/gradcam.png")
    a = ap.parse_args()

    model, ckpt = load_checkpoint(a.ckpt)
    size = ckpt["img_size"]
    img = Image.open(a.image).convert("RGB")
    x = build_transforms(size, train=False)(img).unsqueeze(0)
    cam = GradCAM(model=model, target_layers=[last_conv_layer(model)])
    heat = cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(1)])[0]
    rgb = np.asarray(img.resize((size, size))).astype(np.float32) / 255
    Image.fromarray(show_cam_on_image(rgb, heat, use_rgb=True)).save(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
