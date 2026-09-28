"""Image-level detection dataset retaining every selected reader annotation."""

import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.dicom_reader import read_dicom
from src.data.preprocessing import preprocess


def split_image_ids(image_ids, val_fraction=0.2, seed=42):
    ids = sorted(set(image_ids))
    if len(ids) < 2 or not 0 < val_fraction < 1:
        raise ValueError("Need at least two images and 0 < val_fraction < 1")
    random.Random(seed).shuffle(ids)
    n_val = max(1, min(len(ids) - 1, round(len(ids) * val_fraction)))
    return sorted(ids[n_val:]), sorted(ids[:n_val])


def load_annotations(path):
    annotations = defaultdict(list)
    names = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]
            if not image_id or Path(image_id).name != image_id or "/" in image_id or "\\" in image_id:
                raise ValueError("Invalid image_id")
            class_id = int(row["class_id"])
            if not 0 <= class_id < 14:
                raise ValueError("Subset must contain only abnormal class IDs 0..13")
            label = class_id + 1
            name = row["class_name"].strip()
            if names.setdefault(label, name) != name:
                raise ValueError("Inconsistent class names")
            box = [float(row[k]) for k in ("x_min", "y_min", "x_max", "y_max")]
            annotations[image_id].append((box, label))
    return dict(annotations), names


class VinDrDataset(Dataset):
    def __init__(self, annotations, image_ids, dicom_dir, image_size=512):
        self.annotations = annotations
        self.image_ids = list(image_ids)
        self.dicom_dir = Path(dicom_dir)
        self.image_size = image_size
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("Duplicate image IDs in dataset")
        for image_id in self.image_ids:
            if image_id not in annotations:
                raise ValueError("Image has no annotations")
            if not (self.dicom_dir / f"{image_id}.dicom").is_file():
                raise FileNotFoundError(f"Missing DICOM for image_id {image_id}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.annotations[image_id]
        image = read_dicom(self.dicom_dir / f"{image_id}.dicom")
        image, boxes = preprocess(image, [r[0] for r in records], self.image_size)
        target = {
            "boxes": boxes,
            "labels": torch.tensor([r[1] for r in records], dtype=torch.int64),
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(records), dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
