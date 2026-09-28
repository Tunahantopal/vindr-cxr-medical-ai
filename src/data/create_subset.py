"""Create a reproducible abnormal-image subset. Run from the repository root."""

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COORDS = ("x_min", "y_min", "x_max", "y_max")
SUBSET_SIZE = 500
RANDOM_STATE = 42


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    source = ROOT / "data/raw/train.csv"
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        required = {"image_id", "class_name", "class_id", "rad_id", *COORDS}
        require(fields and required.issubset(fields), "Missing required CSV columns")
        rows = list(reader)
    require(rows, "Input CSV is empty")

    valid = []
    rejected = Counter()
    abnormal_before = Counter()
    class_names = {}
    for line, row in enumerate(rows, start=2):
        require(None not in row and all(row.get(k) is not None for k in fields),
                f"Malformed CSV record at line {line}")
        require(all(row[k].strip() for k in ("image_id", "class_name", "class_id", "rad_id")),
                f"Missing identifier/label at line {line}")
        class_id = int(row["class_id"])
        require(0 <= class_id <= 14, f"Invalid class ID at line {line}")
        name = row["class_name"].strip()
        require(class_names.setdefault(class_id, name) == name, "Inconsistent class names")
        require((class_id == 14) == (name.casefold() == "no finding"),
                f"Inconsistent No finding label at line {line}")
        require(row["image_id"] == row["image_id"].strip(), "Whitespace in image ID")
        if class_id == 14:
            continue
        abnormal_before[name] += 1
        if any(not row[k].strip() for k in COORDS):
            rejected["missing coordinates"] += 1
            continue
        try:
            x_min, y_min, x_max, y_max = map(float, (row[k] for k in COORDS))
        except ValueError:
            rejected["nonnumeric coordinates"] += 1
            continue
        if not all(math.isfinite(v) for v in (x_min, y_min, x_max, y_max)):
            rejected["nonfinite coordinates"] += 1
        elif x_max <= x_min or y_max <= y_min:
            rejected["nonpositive box dimensions"] += 1
        elif x_min < 0 or y_min < 0:
            rejected["negative box origin"] += 1
        else:
            valid.append(row)

    labels = defaultdict(set)
    pools = defaultdict(set)
    for row in valid:
        image_id, name = row["image_id"], row["class_name"].strip()
        labels[image_id].add(name)
        pools[name].add(image_id)
    require(len(labels) >= SUBSET_SIZE, "Fewer than 500 images have valid abnormal boxes")
    original_images = {name: len(ids) for name, ids in pools.items()}
    original_boxes = Counter(row["class_name"].strip() for row in valid)
    rng = random.Random(RANDOM_STATE)
    selected = []
    coverage = Counter()
    # Greedily balance unique-image class counts, breaking ties toward rare classes.
    # Draw one unselected image for the least-covered available class; count ALL
    # its labels and remove it from every pool. This boosts rare classes without
    # assuming mutually exclusive labels or overweighting repeated reader boxes.
    # Sorted candidates plus random_state=42 make choices reproducible.
    for _ in range(SUBSET_SIZE):
        name = min((name for name, ids in pools.items() if ids),
                   key=lambda name: (coverage[name], original_images[name], name))
        image_id = rng.choice(sorted(pools[name]))
        selected.append(image_id)
        for label in sorted(labels[image_id]):
            coverage[label] += 1
            pools[label].remove(image_id)

    selected_ids = set(selected)
    subset = [row for row in valid if row["image_id"] in selected_ids]
    require(len(selected) == len(selected_ids) == SUBSET_SIZE, "Duplicate IDs or wrong subset size")
    require({row["image_id"] for row in subset} == selected_ids,
            "Selected image lacks a valid abnormal box")
    require(set(coverage) == set(original_boxes), "Subset lost a valid abnormal class")
    require(set(original_boxes) == set(abnormal_before), "Box filtering lost an abnormal class")
    for row in subset:
        require(all(row[k].strip() for k in COORDS), "Missing selected bbox coordinate")
        x_min, y_min, x_max, y_max = (float(row[k]) for k in COORDS)
        require(all(math.isfinite(v) for v in (x_min, y_min, x_max, y_max))
                and 0 <= x_min < x_max and 0 <= y_min < y_max, "Invalid selected bbox")

    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "subset_500_image_ids.csv"
    annotations = output / "subset_500_annotations.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id"])
        writer.writerows((image_id,) for image_id in sorted(selected_ids))
    # Retain every valid reader annotation for selected images; no box fusion.
    with annotations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(subset)

    subset_boxes = Counter(row["class_name"].strip() for row in subset)
    print(f"Input: {len(rows)} rows; {len(valid)} valid abnormal boxes; {len(labels)} eligible images")
    print(f"Rejected abnormal rows: {dict(rejected)}")
    print(f"{'Class':25} {'Original boxes':>14} {'Subset boxes':>12} {'Original images':>15} {'Subset images':>13}")
    for name in sorted(original_boxes):
        print(f"{name:25} {original_boxes[name]:14d} {subset_boxes[name]:12d} "
              f"{original_images[name]:15d} {coverage[name]:13d}")
    print(f"Selected unique images: {len(selected_ids)}; annotation rows: {len(subset)}")
    print(f"Class coverage (raw abnormal / valid / subset): "
          f"{len(abnormal_before)} / {len(original_boxes)} / {len(coverage)}")
    print("Validation passed: exactly 500 unique IDs, no duplicates, >=1 valid box per image,")
    print("no missing/nonfinite coordinates or invalid boxes in output; all abnormal classes retained.")
    print(f"Saved {manifest.relative_to(ROOT)} and {annotations.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
