import timm
import torch


def build_model(name: str, pretrained: bool = True) -> torch.nn.Module:
    """Binary classifier: one logit, sigmoid -> P(malignant)."""
    return timm.create_model(name, pretrained=pretrained, num_classes=1)


def load_checkpoint(path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model_name"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
