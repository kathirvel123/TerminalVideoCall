#!/usr/bin/env python3
"""
Terminal ASCII video call over UDP.

Each peer sends webcam frames as JPEG binary chunks.
The remote peer reassembles the image and converts to ASCII locally
(same clarity logic as webcam_ascii.py). Voice is not included yet.
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import threading
import time
from collections import defaultdict

import cv2
import numpy as np

from webcam_ascii import (
    CHAR_ASPECT,
    CLEAR,
    HIDE_CURSOR,
    HOME,
    RESET,
    SHOW_CURSOR,
    frame_to_ascii,
    terminal_grid,
)

MAGIC = b"ASC1"
# magic(4) + frame_id(u32) + chunk_idx(u16) + chunk_count(u16) + payload_len(u16)
HEADER_FMT = "!4sIHHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAX_PAYLOAD = 1200  # stay under typical UDP MTU after IP/UDP headers


class FrameAssembler:
    """Reassemble chunked JPEG frames; keep only the latest complete one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: dict[int, dict[int, bytes]] = defaultdict(dict)
        self._counts: dict[int, int] = {}
        self._latest: bytes | None = None
        self._latest_id = -1
        self.frames_ok = 0
        self.frames_drop = 0

    def push(self, frame_id: int, chunk_idx: int, chunk_count: int, data: bytes) -> None:
        with self._lock:
            if frame_id < self._latest_id:
                return

            self._counts[frame_id] = chunk_count
            bucket = self._chunks[frame_id]
            bucket[chunk_idx] = data

            if len(bucket) == chunk_count:
                parts = [bucket[i] for i in range(chunk_count)]
                payload = b"".join(parts)
                self._latest = payload
                self._latest_id = frame_id
                self.frames_ok += 1
                # Drop older partial frames.
                stale = [fid for fid in self._chunks if fid < frame_id]
                for fid in stale:
                    self._chunks.pop(fid, None)
                    self._counts.pop(fid, None)
                self._chunks.pop(frame_id, None)
                self._counts.pop(frame_id, None)
            elif len(self._chunks) > 8:
                # Too many incomplete frames → drop oldest.
                oldest = min(self._chunks)
                self._chunks.pop(oldest, None)
                self._counts.pop(oldest, None)
                self.frames_drop += 1

    def take_latest(self) -> bytes | None:
        with self._lock:
            data = self._latest
            self._latest = None
            return data


def pack_chunks(frame_id: int, jpeg: bytes) -> list[bytes]:
    total = (len(jpeg) + MAX_PAYLOAD - 1) // MAX_PAYLOAD
    packets: list[bytes] = []
    for idx in range(total):
        start = idx * MAX_PAYLOAD
        piece = jpeg[start : start + MAX_PAYLOAD]
        header = struct.pack(HEADER_FMT, MAGIC, frame_id, idx, total, len(piece))
        packets.append(header + piece)
    return packets


def parse_packet(packet: bytes) -> tuple[int, int, int, bytes] | None:
    if len(packet) < HEADER_SIZE:
        return None
    magic, frame_id, chunk_idx, chunk_count, payload_len = struct.unpack(
        HEADER_FMT, packet[:HEADER_SIZE]
    )
    if magic != MAGIC or chunk_count < 1 or chunk_idx >= chunk_count:
        return None
    payload = packet[HEADER_SIZE : HEADER_SIZE + payload_len]
    if len(payload) != payload_len:
        return None
    return frame_id, chunk_idx, chunk_count, payload


def recv_loop(sock: socket.socket, assembler: FrameAssembler, stop: threading.Event) -> None:
    sock.settimeout(0.5)
    while not stop.is_set():
        try:
            data, _addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break
        parsed = parse_packet(data)
        if parsed is None:
            continue
        frame_id, chunk_idx, chunk_count, payload = parsed
        assembler.push(frame_id, chunk_idx, chunk_count, payload)


def send_loop(
    sock: socket.socket,
    peer: tuple[str, int],
    camera: int,
    width: int,
    quality: int,
    fps: float,
    stop: threading.Event,
) -> None:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"Could not open camera {camera}", file=sys.stderr)
        stop.set()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, max(width, 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max(int(width * 0.75), 480))
    interval = 1.0 / max(fps, 1.0)
    frame_id = 0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    try:
        while not stop.is_set():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if w > width:
                nh = int(h * (width / w))
                frame = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)

            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue

            for packet in pack_chunks(frame_id, buf.tobytes()):
                try:
                    sock.sendto(packet, peer)
                except OSError:
                    stop.set()
                    return
            frame_id = (frame_id + 1) & 0xFFFFFFFF

            sleep_for = interval - (time.perf_counter() - t0)
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        cap.release()


def decode_jpeg(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def parse_peer(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("peer must be host:port, e.g. 192.168.1.10:5000")
    host, port_s = value.rsplit(":", 1)
    try:
        port = int(port_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid peer port") from exc
    if not host or not (1 <= port <= 65535):
        raise argparse.ArgumentTypeError("invalid peer host/port")
    return host, port


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UDP terminal ASCII video call (image binary over UDP, ASCII on receiver)"
    )
    parser.add_argument(
        "--listen",
        type=int,
        required=True,
        help="Local UDP port to bind",
    )
    parser.add_argument(
        "--peer",
        type=parse_peer,
        required=True,
        help="Remote peer as host:port",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Send width in pixels before JPEG (default: 320)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=60,
        help="JPEG quality 1-100 (default: 60)",
    )
    parser.add_argument("--fps", type=float, default=12.0, help="Send FPS cap (default: 12)")
    parser.add_argument("--cols", type=int, default=None, help="Max ASCII columns")
    parser.add_argument("--no-color", action="store_true", help="Grayscale ASCII")
    parser.add_argument("--invert", action="store_true", help="Invert brightness")
    args = parser.parse_args()

    if not sys.stdout.isatty():
        raise SystemExit("Run this in a real terminal (not redirected).")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.listen))
    # Larger buffers help bursty chunked frames.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)

    assembler = FrameAssembler()
    stop = threading.Event()

    threads = [
        threading.Thread(target=recv_loop, args=(sock, assembler, stop), daemon=True),
        threading.Thread(
            target=send_loop,
            args=(sock, args.peer, args.camera, args.width, args.quality, args.fps, stop),
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()

    color = not args.no_color
    last_frame: np.ndarray | None = None
    display_fps = 0.0
    last_fps_t = time.perf_counter()
    frames_drawn = 0

    sys.stdout.write(HIDE_CURSOR + CLEAR)
    sys.stdout.flush()
    print(
        f"Calling {args.peer[0]}:{args.peer[1]} (listening on UDP {args.listen})…",
        file=sys.stderr,
    )

    try:
        while not stop.is_set():
            jpeg = assembler.take_latest()
            if jpeg is not None:
                decoded = decode_jpeg(jpeg)
                if decoded is not None:
                    last_frame = decoded
                    frames_drawn += 1

            cols, rows = terminal_grid(args.cols)
            if last_frame is None:
                placeholder = f"{RESET}Waiting for peer video…  listen:{args.listen}  peer:{args.peer[0]}:{args.peer[1]}"
                sys.stdout.write(HOME + placeholder.ljust(cols) + "\n" + (" " * cols))
                sys.stdout.flush()
                time.sleep(0.05)
                continue

            fit_rows = max(
                1,
                int(cols * CHAR_ASPECT * last_frame.shape[0] / last_frame.shape[1]),
            )
            use_rows = min(rows, fit_rows)
            art = frame_to_ascii(last_frame, cols, use_rows, color=color, invert=args.invert)

            now = time.perf_counter()
            if now - last_fps_t >= 1.0:
                display_fps = frames_drawn / (now - last_fps_t)
                frames_drawn = 0
                last_fps_t = now

            status = (
                f"{RESET}call {args.peer[0]}:{args.peer[1]} | "
                f"{cols}x{use_rows} | {display_fps:.0f} fps | "
                f"ok:{assembler.frames_ok} drop:{assembler.frames_drop} | Ctrl+C quit"
            )
            sys.stdout.write(HOME + art + "\n" + status[:cols].ljust(cols))
            sys.stdout.flush()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        sock.close()
        for t in threads:
            t.join(timeout=1.0)
        sys.stdout.write(SHOW_CURSOR + RESET + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    os.environ.setdefault("TERM", "xterm-256color")
    main()
