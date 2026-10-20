import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(img_size: int, train: bool):
    if train:
        # histology has no canonical orientation, so flips/rotations are safe augmentations
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(0.2, 0.2, 0.15, 0.03),  # mimics stain variation between labs
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class BreakHisDataset(Dataset):
    def __init__(self, splits_csv: str, split: str, img_size: int, train: bool):
        df = pd.read_csv(splits_csv)
        self.df = df[df.split == split].reset_index(drop=True)
        self.tf = build_transforms(img_size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row.path).convert("RGB")
        return self.tf(img), torch.tensor(row.label, dtype=torch.float32), row.patient_id
