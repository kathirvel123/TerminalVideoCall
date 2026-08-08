#!/usr/bin/env python3
"""Live webcam → high-clarity color ASCII in the terminal."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import cv2
import numpy as np

# Dense ramp (dark → light) for sharp detail in the terminal.
ASCII_CHARS = np.array(list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "))

# Terminal cells are taller than wide (~2:1). Tune if your font differs.
CHAR_ASPECT = 0.45

HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
HOME = "\033[H"
CLEAR = "\033[2J"
RESET = "\033[0m"


def frame_to_ascii(frame_bgr: np.ndarray, cols: int, rows: int, color: bool, invert: bool) -> str:
    # OpenCV is BGR; flip for mirror selfie feel.
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)

    # Perceived luminance → character index.
    gray = (
        0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]
    )
    if invert:
        gray = 255.0 - gray

    idx = (gray * ((len(ASCII_CHARS) - 1) / 255.0)).astype(np.int32)
    chars = ASCII_CHARS[idx]

    if not color:
        return "\n".join("".join(row) for row in chars)

    # True-color ANSI (one escape per cell) for the same clarity as the image export.
    r = small[:, :, 0]
    g = small[:, :, 1]
    b = small[:, :, 2]
    lines: list[str] = []
    for y in range(rows):
        parts: list[str] = []
        for x in range(cols):
            parts.append(
                f"\033[38;2;{r[y, x]};{g[y, x]};{b[y, x]}m{chars[y, x]}"
            )
        parts.append(RESET)
        lines.append("".join(parts))
    return "\n".join(lines)


def terminal_grid(max_cols: int | None) -> tuple[int, int]:
    size = shutil.get_terminal_size(fallback=(120, 40))
    # Leave one row for the status line.
    rows = max(10, size.lines - 1)
    cols = max(40, size.columns)
    if max_cols is not None:
        cols = min(cols, max_cols)
    return cols, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam live ASCII art (terminal)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Max ASCII columns (default: full terminal width)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Grayscale ASCII (faster / more compatible)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert brightness (for light terminal themes)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target FPS cap (default: 30)",
    )
    args = parser.parse_args()

    if not sys.stdout.isatty():
        raise SystemExit("Run this in a real terminal (not redirected).")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    # Request a sharper capture when the driver allows it.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    color = not args.no_color
    frame_interval = 1.0 / max(args.fps, 1.0)

    sys.stdout.write(HIDE_CURSOR + CLEAR)
    sys.stdout.flush()

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror
            cols, rows = terminal_grid(args.cols)

            # Fit height from width using character cell aspect, then clamp to terminal.
            fit_rows = max(1, int(cols * CHAR_ASPECT * frame.shape[0] / frame.shape[1]))
            rows = min(rows, fit_rows)

            art = frame_to_ascii(frame, cols, rows, color=color, invert=args.invert)
            elapsed = time.perf_counter() - t0
            fps_now = 1.0 / elapsed if elapsed > 0 else 0.0
            status = f"{RESET}ASCII cam | {cols}x{rows} | {fps_now:.0f} fps | Ctrl+C quit"

            sys.stdout.write(HOME + art + "\n" + status[:cols].ljust(cols))
            sys.stdout.flush()

            sleep_for = frame_interval - (time.perf_counter() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        sys.stdout.write(SHOW_CURSOR + RESET + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    # Help Windows terminals enable ANSI if ever used there.
    os.environ.setdefault("TERM", "xterm-256color")
    main()
