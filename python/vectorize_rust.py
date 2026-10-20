"""Drop-in replacement for ``run_vectorization`` in vectorize.py, backed by the
Rust extension (``pip install arcline_vectorize-*.whl`` or ``maturin develop --release``).

Returns the same ``VectorizationResult`` dict shape as the Python pipeline:
{"low_geometry": [...], "high_geometry": [...], "svg": str, "low_geometry_svg": str}
"""
from __future__ import annotations

import numpy as np
from PIL import Image

import arcline_vectorize  # the compiled extension


def run_vectorization(image_array: np.ndarray) -> dict:
    """Same contract as the original: takes ``np.asarray(pil_image)``."""
    arr = np.ascontiguousarray(image_array)
    h, w = arr.shape[:2]
    if arr.ndim == 2:                       # grayscale -> RGB (what skimage's gray2rgb-free path sees)
        rgba = np.dstack([arr, arr, arr, np.full((h, w), 255, np.uint8)])
        has_alpha = False
    elif arr.shape[2] == 3:                 # RGB: alpha ignored, exactly like the Python pipeline
        rgba = np.dstack([arr, np.full((h, w), 255, np.uint8)])
        has_alpha = False
    elif arr.shape[2] == 4:                 # RGBA: composited over white, like rgba2rgb
        rgba = arr
        has_alpha = True
    else:
        raise ValueError(f"unsupported image shape {arr.shape}")
    if rgba.dtype != np.uint8:
        raise ValueError("expected an 8-bit image")
    # The extension releases the GIL while it runs, so asyncio.to_thread / a
    # ThreadPoolExecutor gives real parallelism across requests.
    return arcline_vectorize.vectorize(rgba.tobytes(), w, h, has_alpha)


def run_vectorization_pil(pil_image: Image.Image) -> dict:
    """Convenience variant straight from a PIL image."""
    has_alpha = pil_image.mode in ("RGBA", "LA", "PA")
    rgba = pil_image.convert("RGBA")
    return arcline_vectorize.vectorize(rgba.tobytes(), rgba.width, rgba.height, has_alpha)
