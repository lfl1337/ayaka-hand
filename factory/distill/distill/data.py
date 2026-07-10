"""Crops -> tensors. Train: RandomResizedCrop+flip+jitter (grip/force are flip-invariant);
val: resize+center-crop, deterministic. ImageNet normalization (torchvision convention)."""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

IMG_SIZE = 160
_NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_TRAIN_TF = v2.Compose([
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(**_NORM),
])
_VAL_TF = v2.Compose([
    v2.Resize(176, antialias=True), v2.CenterCrop(IMG_SIZE),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(**_NORM),
])


class CropsDataset(Dataset):
    def __init__(self, data_root: Path, records: list[dict], train: bool):
        self.root = Path(data_root)
        self.records = records
        self.tf = _TRAIN_TF if train else _VAL_TF

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        img = Image.open(self.root / r["path"]).convert("RGB")
        return self.tf(img), r["grip_idx"], r["force_idx"]


def make_loader(data_root: Path, records: list[dict], train: bool,
                batch_size: int, workers: int) -> DataLoader:
    return DataLoader(CropsDataset(data_root, records, train), batch_size=batch_size,
                      shuffle=train, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      drop_last=train, persistent_workers=workers > 0)
