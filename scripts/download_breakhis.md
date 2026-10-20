# Getting BreakHis

1. Official page: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
   (fill the short request form; you get a download link for `BreaKHis_v1.tar.gz`, ~4 GB).
   Alternative: the Kaggle dataset "breakhis" (search "BreaKHis_v1" on kaggle.com/datasets).
2. Extract so the folder looks like:

```
data/raw/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/400X/SOB_B_A-14-22549AB-400-001.png
data/raw/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/...
```

3. Run `python scripts/make_patient_splits.py --root data/raw/BreaKHis_v1 --mag 400X`.

Cite: Spanhol, F. A., Oliveira, L. S., Petitjean, C., Heutte, L. (2016). A Dataset for Breast Cancer
Histopathological Image Classification. IEEE Transactions on Biomedical Engineering, 63(7), 1455-1462.
