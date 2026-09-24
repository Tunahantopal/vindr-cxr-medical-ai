# VinDr-CXR: Chest X-ray Annotation Analysis

Reproducible exploration of VinBigData/VinDr-CXR annotations with Python 3.12, pandas, NumPy, and Matplotlib. This project examines class imbalance, reader coverage, missing coordinates, and bounding-box geometry to inform a future abnormality detection pipeline.

**Status: annotation EDA completed. Model training and evaluation have not been performed.**

## Motivation

Medical image detection starts with understanding the labels. Multiple radiologists annotate the same image, normal-image rows have no bounding box, and annotation frequencies vary substantially by class. This project makes those assumptions explicit before choosing preprocessing, splitting, or modeling strategies.

## Dataset

The notebook reads the locally supplied VinBigData/VinDr-CXR annotation file at `data/raw/train.csv`. Each row contains an image identifier, class name and ID, radiologist ID, and four bounding-box coordinates. Findings describe this particular file, not every release of VinDr-CXR.

Obtain the dataset separately through its authorized distribution channel and follow its usage terms. Annotations, image identifiers, DICOMs, and patient data are not redistributed. The executed notebook contains aggregate tables and statistical charts only. Images are not required for this analysis.

## Analysis performed

Review the [executed exploration notebook](notebooks/01_dataset_exploration.ipynb) for:

- Shape, unique X-rays, column types, missing values, and exact duplicates.
- Class counts at annotation and unique-image levels, with comparison charts.
- No Finding frequency and checks for mixed normal/abnormal labels.
- Annotation volume and image coverage per radiologist.
- Box completeness, finite coordinates, positive dimensions, and nonnegative origins.
- Width, height, and area statistics, with linear and logarithmic area histograms.
- Annotation and box counts per image, with aggregate distributions.

## Actual EDA findings

| Measure | Result |
|---|---:|
| Annotation table | 67,914 rows × 8 columns |
| Unique X-rays | 15,000 |
| Labels | 14 abnormality classes + No Finding |
| Radiologists | 17; exactly 3 distinct readers per image |
| No Finding rows | 31,818 (46.85% of rows) |
| Images with only No Finding | 10,606 (70.71% of images) |
| Images with abnormal annotations | 4,394 (29.29%) |
| Images mixing No Finding and abnormal labels | 0 |
| Valid abnormal annotation boxes | 36,096 |
| Exact duplicate rows | 0 |
| Annotation rows per image | Mean 4.53; median 3; maximum 57 |

Each coordinate column has 31,818 missing values, all on No Finding rows. No abnormal rows have missing coordinates or fail the notebook's geometry checks. Identifier and label columns have no missing values.

**Aortic enlargement (7,162 annotation rows)** is the most frequent abnormal label; **Pneumothorax (226)** is the least frequent, a **31.7:1** ratio. Counts include multiple readers and do not measure disease prevalence or unique lesions.

| Box measurement | Mean | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Width (pixels) | 440.94 | 323 | 11 | 2,938 |
| Height (pixels) | 391.40 | 320 | 3 | 2,803 |
| Area (pixels²) | 218,415.71 | 106,471 | 180 | 4,575,318 |

The area distribution has a long upper tail. Pixel areas cannot establish relative or physical lesion size without image dimensions. Three readers per image also means a deliberate cross-reader aggregation policy is needed before training.

## Project structure

```text
vindr-cxr-medical-ai/
├── notebooks/
│   └── 01_dataset_exploration.ipynb  # Executed EDA; aggregate outputs
├── docs/
│   ├── decisions.md                 # Placeholder for future decisions
│   └── experiments.md               # Placeholder for future experiments
├── data/raw/train.csv               # Local input; ignored, not distributed
├── .gitignore
├── requirements.txt                # Direct EDA/notebook dependencies
└── README.md
```

The local `src/`, `configs/`, and `tests/` directories are empty placeholders, not implemented training infrastructure.

## Setup and execution

Use **Python 3.12**. From the repository root on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name vindr-cxr --display-name "VinDr-CXR (Python 3.12)"
.\.venv\Scripts\python.exe -m jupyter lab
```

On macOS/Linux, create the environment with `python3.12 -m venv .venv` and substitute `.venv/bin/python` for the Windows executable in subsequent commands.

Place the actual annotation **file** at `data/raw/train.csv`, open the notebook, select the VinDr-CXR kernel, and run all cells. Paths support launch from the repository root or `notebooks/`, without machine-specific absolute paths.

For headless execution from the repository root:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_dataset_exploration.ipynb --ExecutePreprocessor.kernel_name=vindr-cxr --ExecutePreprocessor.timeout=120
```

This refreshes aggregate outputs in place. Direct dependencies are pinned to the versions used for this analysis; pip resolves transitive dependencies rather than maintaining an environment dump.

## Limitations

- Annotation-only analysis: no DICOM inspection, image-quality review, or clinical validation.
- No Finding is an annotation label, not proof of absence of disease.
- All reader annotations are retained; overlapping boxes are not spatially deduplicated or fused.
- Image dimensions are absent from the CSV; box bounds and normalized areas are unverified.
- Patient linkage, leakage assessment, and train/validation/test splits are not implemented.
- No trained models, performance claims, or deployment artifacts. This is a research portfolio project, not a diagnostic tool.
- `.gitignore` excludes data, image exports, weights, environments, checkpoints, and secrets from ordinary Git adds. It cannot prevent forced adds or remove previously tracked files; inspect staged files before committing.

## Roadmap

1. Validate DICOM dimensions and box bounds; document preprocessing.
2. Define and compare cross-reader aggregation policies.
3. Create leakage-aware splits using patient identifiers if available; document any image-level fallback.
4. Train a reproducible detection baseline.
5. Report per-class detection metrics, rare-class behavior, and held-out error analysis.
6. Record experiment results and model limitations before inference packaging.
