"""Train the benign/malignant classifier.

Usage: python -m histo_agent.vision.train --config configs/vision.yaml
"""
import argparse
import json
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from histo_agent.vision.dataset import BreakHisDataset
from histo_agent.vision.model import build_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels, pids = [], [], []
    for x, y, pid in loader:
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            p = torch.sigmoid(model(x.to(device)).squeeze(1))
        probs += p.float().cpu().tolist()
        labels += y.tolist()
        pids += list(pid)
    return np.array(probs), np.array(labels), pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vision.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    seed_all(t["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)

    train_ds = BreakHisDataset(d["splits_csv"], "train", d["img_size"], train=True)
    val_ds = BreakHisDataset(d["splits_csv"], "val", d["img_size"], train=False)
    train_dl = DataLoader(train_ds, t["batch_size"], shuffle=True, num_workers=d["num_workers"], pin_memory=True)
    val_dl = DataLoader(val_ds, t["batch_size"] * 2, num_workers=d["num_workers"])

    model = build_model(m["name"], m["pretrained"]).to(device)
    # class imbalance: BreakHis is about 69% malignant, so weight the positive class down
    pos = train_ds.df.label.mean()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=...)

    mlflow.set_experiment("histo-agent-vision")
    best_auc, bad = -1.0, 0
    with mlflow.start_run():
        mlflow.log_params({**m, **t, "img_size": d["img_size"], "train_images": len(train_ds)})
        for epoch in range(t["epochs"]):
            model.train()
            total = 0.0
            for x, y, _ in tqdm(train_dl, desc=f"epoch {epoch + 1}/{t['epochs']}"):
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, enabled=scaler.is_enabled()):
                    loss = loss_fn(model(x).squeeze(1), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item() * len(y)
            sched.step()

            probs, labels, _ = predict(model, val_dl, device)
            auc = roc_auc_score(labels, probs)
            mlflow.log_metrics({"train_loss": total / len(train_ds), "val_auroc": auc}, step=epoch)
            print(f"epoch {epoch + 1}: loss {total / len(train_ds):.4f}  val AUROC {auc:.4f}")
            if auc > best_auc:
                best_auc, bad = auc, 0
                torch.save({"model_name": m["name"], "img_size": d["img_size"], "state_dict": model.state_dict()},
                           out / "best.pt")
            else:
                bad += 1
                if bad >= t["early_stop_patience"]:
                    print("early stopping")
                    break
        mlflow.log_metric("best_val_auroc", best_auc)

    json.dump({"best_val_auroc": best_auc}, open(out / "train_summary.json", "w"), indent=2)
    print(f"best val AUROC {best_auc:.4f}; checkpoint at {out / 'best.pt'}")


if __name__ == "__main__":
    main()
