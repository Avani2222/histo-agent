"""Pick the decision threshold on validation, then report test metrics at image and patient level.

Usage: python -m histo_agent.vision.evaluate --config configs/vision.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from histo_agent.vision.dataset import BreakHisDataset
from histo_agent.vision.model import load_checkpoint
from histo_agent.vision.train import predict


def threshold_for_sensitivity(labels, probs, target):
    """Highest threshold whose sensitivity on validation is at least `target`."""
    fpr, tpr, thr = roc_curve(labels, probs)
    ok = np.where(tpr >= target)[0]
    return float(min(thr[ok[0]], 1.0)) if len(ok) else 0.5


def metrics(labels, probs, thr):
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(labels, probs)),
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "accuracy": (tp + tn) / len(labels), "n": int(len(labels))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vision.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    d, out = cfg["data"], Path(cfg["output"]["dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(out / "best.pt", device)

    res = {}
    for split in ["val", "test"]:
        ds = BreakHisDataset(d["splits_csv"], split, d["img_size"], train=False)
        res[split] = predict(model, DataLoader(ds, 32, num_workers=d["num_workers"]), device)

    thr = threshold_for_sensitivity(res["val"][1], res["val"][0], cfg["eval"]["target_sensitivity"])
    probs, labels, pids = res["test"]
    image_level = metrics(labels, probs, thr)
    # patient-level score = mean probability over that patient's images
    pdf = pd.DataFrame({"pid": pids, "p": probs, "y": labels}).groupby("pid").mean()
    patient_level = metrics(pdf.y.values.astype(int), pdf.p.values, thr)

    report = {"threshold": thr, "target_sensitivity": cfg["eval"]["target_sensitivity"],
              "test_image_level": image_level, "test_patient_level": patient_level}
    json.dump({"threshold": thr}, open(out / "threshold.json", "w"))
    json.dump(report, open(out / "test_metrics.json", "w"), indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
