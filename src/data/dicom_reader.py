"""Decode a single grayscale DICOM into float32 [0, 1] pixels."""

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut


def read_dicom(path):
    ds = pydicom.dcmread(path)
    photometric = str(ds.PhotometricInterpretation)
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError(f"Unsupported photometric interpretation: {photometric}")
    raw = ds.pixel_array
    if raw.ndim != 2:
        raise ValueError("Expected a single-frame grayscale DICOM")
    valid = np.ones(raw.shape, dtype=bool)
    if "PixelPaddingValue" in ds:
        low = float(ds.PixelPaddingValue)
        high = float(ds.get("PixelPaddingRangeLimit", low))
        valid &= ~((raw >= min(low, high)) & (raw <= max(low, high)))
    pixels = np.asarray(apply_modality_lut(raw, ds), dtype=np.float32)
    if not np.isfinite(pixels).all() or not valid.any():
        raise ValueError("DICOM has nonfinite pixels or contains only padding")
    low, high = pixels[valid].min(), pixels[valid].max()
    # Per-image min/max after modality rescaling; no VOI/windowing or equalization.
    if high > low:
        pixels = np.clip((pixels - low) / (high - low), 0, 1)
        if photometric == "MONOCHROME1":
            pixels = 1.0 - pixels
    else:
        pixels = np.zeros_like(pixels)
    pixels[~valid] = 0
    return np.ascontiguousarray(pixels, dtype=np.float32)
