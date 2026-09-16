#!/usr/bin/env python3
"""
media_renderer.py — Convert images and animated GIFs to ANSI ASCII art.

Reuses the core frame_to_ascii logic from webcam_ascii.py and adds:
  - Static image rendering (JPEG, PNG, etc.)
  - Animated GIF rendering (returns a list of ASCII frames + durations)
  - A helper to play a GIF in the terminal inline
"""

from __future__ import annotations

import io
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageSequence

from webcam_ascii import frame_to_ascii, terminal_grid, CHAR_ASPECT, RESET


def image_bytes_to_ascii(
    data: bytes,
    max_cols: int | None = None,
    color: bool = True,
    invert: bool = False,
) -> str:
    """
    Convert raw image bytes (JPEG, PNG, etc.) to an ANSI ASCII string.
    Fits the image into the current terminal dimensions.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return "[Could not decode image]"

    cols, rows = terminal_grid(max_cols)
    # Keep aspect ratio
    h, w = bgr.shape[:2]
    fit_rows = max(1, int(cols * CHAR_ASPECT * h / w))
    use_rows = min(rows // 2, fit_rows)  # use at most half the terminal for inline

    return frame_to_ascii(bgr, cols, use_rows, color=color, invert=invert)


def gif_bytes_to_ascii_frames(
    data: bytes,
    max_cols: int | None = None,
    color: bool = True,
    invert: bool = False,
) -> List[Tuple[str, float]]:
    """
    Convert an animated GIF to a list of (ascii_string, duration_seconds) tuples.
    If data is not a valid GIF, falls back to treating it as a static image.
    """
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        return [(image_bytes_to_ascii(data, max_cols, color, invert), 0.1)]

    if not getattr(img, "is_animated", False):
        # Static image or single-frame GIF — treat as plain image
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        return [(image_bytes_to_ascii(buf.getvalue(), max_cols, color, invert), 0.0)]

    cols, rows = terminal_grid(max_cols)
    use_rows = max(1, rows // 2)

    frames: List[Tuple[str, float]] = []
    for frame in ImageSequence.Iterator(img):
        duration_ms = frame.info.get("duration", 100)
        rgb = frame.convert("RGB")
        bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        fit_rows = max(1, int(cols * CHAR_ASPECT * h / w))
        actual_rows = min(use_rows, fit_rows)
        art = frame_to_ascii(bgr, cols, actual_rows, color=color, invert=invert)
        frames.append((art, duration_ms / 1000.0))

    return frames


def detect_mime(data: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "application/octet-stream"


async def render_media_inline(
    data: bytes,
    mime: str | None = None,
    color: bool = True,
    invert: bool = False,
) -> List[Tuple[str, float]]:
    """
    High-level helper: given raw bytes, return a list of (ascii_art, duration) tuples.
    - Static images return a single-item list with duration 0.
    - GIFs return multiple items, one per frame.
    """
    if mime is None:
        mime = detect_mime(data)

    if mime == "image/gif":
        return gif_bytes_to_ascii_frames(data, color=color, invert=invert)
    else:
        art = image_bytes_to_ascii(data, color=color, invert=invert)
        return [(art, 0.0)]
