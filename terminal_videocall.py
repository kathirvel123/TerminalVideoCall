#!/usr/bin/env python3
"""
Terminal ASCII video call over UDP with voice.

Video: each peer sends webcam frames as JPEG binary chunks; the remote peer
reassembles the image and converts to ASCII locally (same clarity logic as
webcam_ascii.py).

Voice: mic audio captured with sounddevice, sent as raw PCM (int16 mono)
UDP packets, reordered in a small jitter buffer and played back on the
speaker. If audio is unavailable the call silently continues video-only.
"""

from __future__ import annotations

import argparse
import os
import queue
import socket
import struct
import sys
import threading
import time
from collections import defaultdict

try:
    import sounddevice as sd
except (ImportError, OSError):
    # voice is simply disabled if sounddevice (or its PortAudio lib) is missing
    sd = None

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
PT_VIDEO = 0
PT_AUDIO = 1

# magic(4) + ptype(u8) + seq(u32) + chunk_idx(u16) + chunk_count(u16) + payload_len(u16)
HEADER_FMT = "!4sBIHHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAX_PAYLOAD = 1200  # stay under typical UDP MTU after IP/UDP headers

AUDIO_SAMPLERATE = 48000
AUDIO_BLOCK_MS = 10  # one voice packet per 10 ms → 960 bytes, safe under MTU


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


def pack_chunks(seq: int, jpeg: bytes, ptype: int = PT_VIDEO) -> list[bytes]:
    total = (len(jpeg) + MAX_PAYLOAD - 1) // MAX_PAYLOAD
    packets: list[bytes] = []
    for idx in range(total):
        start = idx * MAX_PAYLOAD
        piece = jpeg[start : start + MAX_PAYLOAD]
        header = struct.pack(HEADER_FMT, MAGIC, ptype, seq, idx, total, len(piece))
        packets.append(header + piece)
    return packets


def pack_audio(seq: int, pcm: bytes) -> bytes:
    return struct.pack(HEADER_FMT, MAGIC, PT_AUDIO, seq, 0, 1, len(pcm)) + pcm


def parse_packet(packet: bytes) -> tuple[int, int, int, int, bytes] | None:
    if len(packet) < HEADER_SIZE:
        return None
    magic, ptype, seq, chunk_idx, chunk_count, payload_len = struct.unpack(
        HEADER_FMT, packet[:HEADER_SIZE]
    )
    if magic != MAGIC or chunk_count < 1 or chunk_idx >= chunk_count:
        return None
    payload = packet[HEADER_SIZE : HEADER_SIZE + payload_len]
    if len(payload) != payload_len:
        return None
    return ptype, seq, chunk_idx, chunk_count, payload


def recv_loop(
    sock: socket.socket,
    assembler: FrameAssembler,
    audio_rx: "AudioReceiver | None",
    stop: threading.Event,
) -> None:
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
        ptype, seq, chunk_idx, chunk_count, payload = parsed
        if ptype == PT_VIDEO:
            assembler.push(seq, chunk_idx, chunk_count, payload)
        elif ptype == PT_AUDIO and audio_rx is not None:
            audio_rx.push(seq, payload)


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


class AudioSender:
    """Capture mic via sounddevice and ship raw PCM (int16 mono) over UDP."""

    def __init__(
        self,
        sock: socket.socket,
        peer: tuple[str, int],
        samplerate: int,
        blocksize: int,
        device: int | None,
        stop: threading.Event,
    ) -> None:
        self.sock = sock
        self.peer = peer
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.stop = stop
        self._seq = 0
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=256)

    def _mic_callback(self, indata, _frames, _time_info, _status) -> None:
        blob = np.ascontiguousarray(indata).tobytes()
        try:
            self._queue.put_nowait(blob)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop oldest if we fall behind
                self._queue.put_nowait(blob)
            except queue.Empty:
                pass

    def run(self) -> None:
        with sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=self._mic_callback,
        ):
            while not self.stop.is_set():
                try:
                    blob = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self.sock.sendto(pack_audio(self._seq, blob), self.peer)
                except OSError:
                    break
                self._seq = (self._seq + 1) & 0xFFFFFFFF


class AudioReceiver:
    """Reorder received PCM in a small jitter buffer and play on the speaker."""

    def __init__(
        self,
        samplerate: int,
        blocksize: int,
        device: int | None,
        stop: threading.Event,
        max_buffered: int = 200,
    ) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.stop = stop
        self._max = max_buffered
        self._lock = threading.Lock()
        self._buf: dict[int, bytes] = {}
        self._next = 0
        self._initialized = False

    def push(self, seq: int, pcm: bytes) -> None:
        with self._lock:
            if not self._initialized:
                self._initialized = True
                self._next = seq
                self._buf[seq] = pcm
                return
            if seq < self._next - 1:
                return  # already played / stale
            self._buf[seq] = pcm
            if len(self._buf) > self._max:
                for old in sorted(self._buf)[: len(self._buf) - self._max]:
                    del self._buf[old]

    def _play_callback(self, outdata, frames, _time_info, _status) -> None:
        with self._lock:
            if self._next in self._buf:
                data = self._buf.pop(self._next)
                self._next += 1
            elif self._buf:
                smallest = min(self._buf)
                if smallest - self._next > 20:  # big gap → resync past it
                    self._next = smallest
                    data = self._buf.pop(self._next)
                    self._next += 1
                else:
                    data = None
                    self._next += 1  # one slot of silence, keep moving
            else:
                data = None
                self._next += 1
        if data is None:
            outdata.fill(0)
            return
        arr = np.frombuffer(data, dtype=np.int16)
        if arr.size >= frames:
            outdata[:, 0] = arr[:frames]
        else:
            outdata[:, 0] = np.concatenate(
                (arr, np.zeros(frames - arr.size, dtype=np.int16))
            )

    def run(self) -> None:
        with sd.RawOutputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=self._play_callback,
        ):
            while not self.stop.is_set():
                time.sleep(0.1)


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
        description="UDP terminal ASCII video call (image + voice over UDP)"
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
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable voice; video only",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=AUDIO_SAMPLERATE,
        help=f"Audio sample rate Hz (default: {AUDIO_SAMPLERATE})",
    )
    parser.add_argument(
        "--mic",
        type=int,
        default=None,
        help="Input (mic) sounddevice device index",
    )
    parser.add_argument(
        "--speaker",
        type=int,
        default=None,
        help="Output (speaker) sounddevice device index",
    )
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
        threading.Thread(
            target=send_loop,
            args=(sock, args.peer, args.camera, args.width, args.quality, args.fps, stop),
            daemon=True,
        ),
    ]

    audio_ok = False
    audio_rx: AudioReceiver | None = None
    if not args.no_audio and sd is not None:
        blocksize = max(1, int(args.samplerate * AUDIO_BLOCK_MS / 1000))

        def _audio_guard(name: str, fn) -> threading.Thread:
            def wrapper() -> None:
                try:
                    fn()
                except Exception as exc:  # noqa: BLE001 - keep the call alive
                    if not stop.is_set():
                        print(
                            f"Audio {name} failed ({exc}); continuing video-only.",
                            file=sys.stderr,
                        )

            return threading.Thread(target=wrapper, daemon=True)

        tx = AudioSender(sock, args.peer, args.samplerate, blocksize, args.mic, stop)
        rx = AudioReceiver(args.samplerate, blocksize, args.speaker, stop)
        audio_rx = rx
        audio_ok = True
        threads.append(_audio_guard("mic", tx.run))
        threads.append(_audio_guard("speaker", rx.run))
    elif args.no_audio:
        print("Voice: off (--no-audio)", file=sys.stderr)
    else:
        print("Voice: off (sounddevice not installed)", file=sys.stderr)

    threads.insert(
        0,
        threading.Thread(
            target=recv_loop, args=(sock, assembler, audio_rx, stop), daemon=True
        ),
    )
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
        f"Calling {args.peer[0]}:{args.peer[1]} (listening on UDP {args.listen})… "
        + ("voice:on" if audio_ok else "voice:off"),
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
                f"voice:{'on' if audio_ok else 'off'} | "
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
