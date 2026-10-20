"""Build patient-level train/val/test splits for BreakHis.

BreakHis file names look like  SOB_B_A-14-22549AB-400-001.png
  SOB_<B|M>_<subtype> - <year> - <slide id> - <magnification> - <seq>
The patient (slide) id is "<year>-<slide id>". Splitting by patient prevents the same patient
appearing in train and test, which otherwise inflates every metric.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SUBTYPES = {
    "A": "adenosis", "F": "fibroadenoma", "PT": "phyllodes_tumor", "TA": "tubular_adenoma",
    "DC": "ductal_carcinoma", "LC": "lobular_carcinoma", "MC": "mucinous_carcinoma", "PC": "papillary_carcinoma",
}


def parse(path: Path) -> dict:
    stem = path.stem  # SOB_B_A-14-22549AB-400-001
    head, year, slide, mag, _seq = stem.split("-")
    _, cls, sub = head.split("_")
    return {
        "path": str(path.as_posix()),
        "patient_id": f"{year}-{slide}",
        "label": 1 if cls == "M" else 0,  # 1 = malignant
        "subtype": SUBTYPES.get(sub, sub),
        "magnification": f"{mag}X",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to BreaKHis_v1")
    ap.add_argument("--mag", default="400X", help="40X | 100X | 200X | 400X | all")
    ap.add_argument("--out", default="data/processed/splits.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(Path(args.root).rglob("*.png"))
    if not files:
        raise SystemExit(f"No .png files under {args.root}. See scripts/download_breakhis.md")
    df = pd.DataFrame(parse(p) for p in files)
    if args.mag != "all":
        df = df[df.magnification == args.mag].reset_index(drop=True)

    # ~70/15/15 by patient, stratified by label. 82 patients total, so splits are small; that is expected.
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=args.seed)
    folds = list(sgkf.split(df, df.label, groups=df.patient_id))
    df["split"] = "train"
    df.loc[folds[0][1], "split"] = "test"
    df.loc[folds[1][1], "split"] = "val"

    # sanity: no patient in more than one split
    leak = df.groupby("patient_id").split.nunique()
    assert (leak == 1).all(), "patient leakage between splits"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = df.groupby(["split", "label"]).agg(images=("path", "size"), patients=("patient_id", "nunique"))
    print(summary)
    print(f"wrote {args.out} ({len(df)} images, {df.patient_id.nunique()} patients)")


if __name__ == "__main__":
    main()
