"""Small Faster R-CNN baseline. Run as python -m src.training.train."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.data.dataset import VinDrDataset, collate_fn, load_annotations, split_image_ids


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_loaders(config):
    annotations, names = load_annotations(config["annotations"])
    with open(config["manifest"], newline="", encoding="utf-8-sig") as handle:
        manifest = [r["image_id"] for r in csv.DictReader(handle)]
    if (len(manifest) != config["expected_images"] or len(set(manifest)) != len(manifest)
            or set(manifest) != set(annotations)):
        raise ValueError("Subset manifest/annotations mismatch or incorrect image count")
    if set(names) != set(range(1, 15)):
        raise ValueError("Expected all 14 abnormal classes in the subset")
    train_ids, val_ids = split_image_ids(manifest, config["val_fraction"], config["seed"])
    if set(train_ids) & set(val_ids):
        raise ValueError("Image leakage between train and validation")
    datasets = [VinDrDataset(annotations, ids, config["dicom_dir"], config["image_size"])
                for ids in (train_ids, val_ids)]
    loaders = [DataLoader(ds, batch_size=config["batch_size"], shuffle=(i == 0),
                          num_workers=config["num_workers"], collate_fn=collate_fn,
                          worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(config["seed"] + i))
               for i, ds in enumerate(datasets)]
    return loaders[0], loaders[1], {"train": train_ids, "validation": val_ids}, names


def build_model(image_size=512, pretrained=True):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None,
                                   min_size=image_size, max_size=image_size)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 15)
    return model


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    totals = {}
    count = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        losses = model(images, targets)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        n = len(images)
        count += n
        for key, value in {**losses, "loss": loss}.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * n
    return {f"train_{key}": value / count for key, value in totals.items()}


@torch.inference_mode()
def evaluate(model, loader, device):
    # Optional hook boundary: replace this evaluator to test reader-box fusion later.
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    model.eval()
    for images, targets in loader:
        predictions = model([image.to(device) for image in images])
        metric.update([{k: v.cpu() for k, v in p.items()} for p in predictions],
                      [{k: t[k].cpu() for k in ("boxes", "labels")} for t in targets])
    result = metric.compute()
    return {f"val_{k}": float(result[k]) for k in ("map", "map_50", "map_75", "mar_100")}


def run(config):
    if config["epochs"] < 1:
        raise ValueError("epochs must be positive")
    seed_everything(config["seed"])
    requested = config["device"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if requested == "auto" else requested)
    train_loader, val_loader, splits, names = build_loaders(config)
    output = Path(config["output_dir"])
    # Keep IDs, metrics and weights inside the already ignored checkpoint tree.
    checkpoint_root = Path("checkpoints").resolve()
    if not output.resolve().is_relative_to(checkpoint_root):
        raise ValueError("output_dir must be under checkpoints/ (ignored)")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use a new output_dir under checkpoints/ for each run")
    (output / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    model = build_model(config["image_size"]).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=config["learning_rate"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"])
    history, best = [], -math.inf
    for epoch in range(1, config["epochs"] + 1):
        record = {"epoch": epoch, **train_one_epoch(model, train_loader, optimizer, device),
                  **evaluate(model, val_loader, device)}
        if not all(math.isfinite(v) for v in record.values()):
            raise RuntimeError("Nonfinite epoch metrics")
        history.append(record)
        improved = record["val_map"] > best
        best = max(best, record["val_map"])
        checkpoint = {"epoch": epoch, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "best_map": best,
                      "config": config, "class_names": names, "splits": splits,
                      "history": history,
                      "versions": {"torch": str(torch.__version__),
                                   "torchvision": str(torchvision.__version__)}}
        # Atomic replacement avoids a partially written latest checkpoint.
        temporary = output / "pending.pt"
        torch.save(checkpoint, temporary)
        temporary.replace(output / "latest.pt")
        if improved:
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best.pt")
        (output / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run(config)


if __name__ == "__main__":
    main()
