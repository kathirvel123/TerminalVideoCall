#!/usr/bin/env python3
"""
chat_app.py — Simple CLI chat client (no Textual).

All the real logic is here: WebRTC, signaling, storage, media rendering.
UI will be layered on top once the logic is solid.

Run:
    python chat_app.py
    SIGNALING_URL=wss://your-app.onrender.com python chat_app.py

Commands:
    /chat <peer_id>    — set active peer
    /call <peer_id>    — start ASCII video call
    /hangup            — end current call
    /send <filepath>   — send an image or GIF
    /name <username>   — change your display name
    /myid              — print your user ID
    /history           — show chat history with active peer
    /quit              — exit
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np

from storage import StorageManager
from webrtc_engine import WebRTCEngine
from media_renderer import render_media_inline
from webcam_ascii import frame_to_ascii, terminal_grid, CHAR_ASPECT

# ── ANSI helpers ──────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
CLEAR   = "\033[2J\033[H"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
HOME    = "\033[H"

def c(text, *codes):
    return "".join(codes) + str(text) + RESET


# ── Config ────────────────────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    if url.startswith("https://"):
        return "wss://" + url[len("https://"):]
    if url.startswith("http://"):
        return "ws://" + url[len("http://"):]
    return url

SIGNALING_URL = _normalize_url(os.getenv("SIGNALING_URL", "ws://localhost:8000"))
CAMERA_INDEX  = int(os.getenv("CAMERA_INDEX", "0"))


# ── State ─────────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.active_peer: Optional[str] = None
        self.in_call: bool = False
        self.call_peer: Optional[str] = None
        self.call_frame: Optional[np.ndarray] = None


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_banner(user_id: str, username: str):
    print(c(f"\n  ⬡  Terminal P2P Chat", BOLD, CYAN))
    print(c(f"  User: {username}  |  ID: {user_id}", DIM))
    print(c(f"  Server: {SIGNALING_URL}", DIM))
    print(c("  " + "─" * 50, DIM))
    print(c("  /myid  /chat <id>  /call <id>  /send <file>  /quit", DIM))
    print()


def print_history(history: list, my_id: str):
    if not history:
        print(c("  (no messages yet)", DIM))
        return
    for msg in history:
        sender = msg["sender_id"]
        ts = msg.get("timestamp", "")[:16]
        content = msg["content"]
        mtype = msg.get("message_type", "text")
        if mtype == "image_ascii":
            line = f"{c(ts, DIM)}  {c(sender[:8], YELLOW)}: {c('[media]', DIM)}"
        elif sender == my_id:
            line = f"{c(ts, DIM)}  {c('You', GREEN, BOLD)}: {content}"
        else:
            line = f"{c(ts, DIM)}  {c(sender[:8], MAGENTA, BOLD)}: {content}"
        print(line)


def print_msg(sender: str, content: str, my_id: str):
    ts = datetime.now().strftime("%H:%M")
    if sender == my_id:
        print(f"\r{c(ts, DIM)}  {c('You', GREEN, BOLD)}: {content}")
    else:
        print(f"\r{c(ts, DIM)}  {c(sender[:8], MAGENTA, BOLD)}: {content}")


# ── Command handler ───────────────────────────────────────────────────────────

async def handle_command(
    raw: str,
    state: AppState,
    engine: WebRTCEngine,
    storage: StorageManager,
) -> bool:
    """Returns True if the app should keep running."""
    parts = raw.strip().split(maxsplit=1)
    cmd  = parts[0].lower()
    arg  = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit", "/q"):
        return False

    elif cmd == "/myid":
        print(c(f"  Your User ID: {storage.get_user_id()}", CYAN))

    elif cmd == "/help":
        print(c("""
  Commands:
    /chat <peer_id>    — open chat with a peer
    /call <peer_id>    — start ASCII video call
    /hangup            — end current call
    /send <filepath>   — send image or GIF to active peer
    /history           — show chat history with active peer
    /name <username>   — change your display name
    /myid              — show your User ID
    /quit              — exit
""", DIM))

    elif cmd == "/name":
        if not arg:
            print(c("  Usage: /name <new_username>", RED))
        else:
            storage.update_username(arg)
            print(c(f"  Username updated to: {arg}", GREEN))

    elif cmd == "/chat":
        if not arg:
            print(c("  Usage: /chat <peer_id>", RED))
        else:
            state.active_peer = arg
            print(c(f"  Active peer set to: {arg}", CYAN))
            history = storage.get_chat_history(arg, limit=20)
            print(c(f"  ─── Last {len(history)} messages ───", DIM))
            print_history(history, storage.get_user_id())
            print(c(f"  ─────────────────────────────", DIM))

    elif cmd == "/history":
        if not state.active_peer:
            print(c("  No active peer. Use /chat <peer_id> first.", RED))
        else:
            history = storage.get_chat_history(state.active_peer, limit=50)
            print_history(history, storage.get_user_id())

    elif cmd == "/call":
        peer = arg or state.active_peer
        if not peer:
            print(c("  Usage: /call <peer_id>", RED))
        else:
            state.in_call = True
            state.call_peer = peer
            state.active_peer = peer
            print(c(f"  Calling {peer}... (video will render here)", YELLOW))
            try:
                await engine.start_call(peer, CAMERA_INDEX)
            except Exception as e:
                print(c(f"  Call error: {e}", RED))
                state.in_call = False

    elif cmd == "/hangup":
        if state.call_peer:
            await engine.end_call(state.call_peer)
            state.in_call = False
            state.call_frame = None
            state.call_peer = None
            print(c("  Call ended.", YELLOW))
        else:
            print(c("  No active call.", DIM))

    elif cmd == "/send":
        if not arg:
            print(c("  Usage: /send <filepath>", RED))
        elif not state.active_peer:
            print(c("  No active peer. Use /chat <peer_id> first.", RED))
        elif not os.path.exists(arg):
            print(c(f"  File not found: {arg}", RED))
        else:
            with open(arg, "rb") as f:
                data = f.read()
            ext  = arg.rsplit(".", 1)[-1].lower()
            mime = "image/gif" if ext == "gif" else f"image/{ext}"
            print(c(f"  Sending {os.path.basename(arg)}...", DIM))
            try:
                await engine.send_media(state.active_peer, data, mime)
                # Render locally for confirmation
                frames = await render_media_inline(data, mime)
                print(frames[0][0])
                storage.save_message(
                    state.active_peer, storage.get_user_id(),
                    "image_ascii", f"[sent {os.path.basename(arg)}]"
                )
                print(c(f"  ✓ Sent.", GREEN))
            except Exception as e:
                print(c(f"  Send error: {e}", RED))

    else:
        print(c(f"  Unknown command: {cmd}  (type /help)", RED))

    return True


# ── Video call render loop ────────────────────────────────────────────────────

async def video_render_loop(state: AppState):
    """While in a call, render the latest frame as ASCII at ~10 fps."""
    while True:
        await asyncio.sleep(0.1)
        if state.in_call and state.call_frame is not None:
            cols, rows = terminal_grid(None)
            frame = state.call_frame
            h, w = frame.shape[:2]
            fit_rows = max(1, int(cols * CHAR_ASPECT * h / w))
            use_rows = min(rows - 3, fit_rows)
            art = frame_to_ascii(frame, cols, use_rows, color=True, invert=False)
            status = c(
                f"  📞 In call with {state.call_peer}  |  /hangup to end",
                YELLOW
            )
            sys.stdout.write(HOME + art + "\n" + status + "\n")
            sys.stdout.flush()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    storage = StorageManager()

    # First-run setup
    username = storage.get_username()
    user_id  = storage.get_user_id()
    if username.startswith("User_"):
        print(c("\n  Welcome to Terminal P2P Chat!", BOLD, CYAN))
        print(c(f"  Your User ID: {user_id}", CYAN))
        new_name = input("  Choose a username: ").strip()
        if new_name:
            storage.update_username(new_name)
            username = new_name

    state = AppState()

    # ── WebRTC callbacks ──────────────────────────────────────────────────────

    async def on_text(peer_id: str, text: str):
        storage.save_message(peer_id, peer_id, "text", text)
        print_msg(peer_id, text, user_id)
        # Reprint prompt
        peer_label = f"({state.active_peer}) " if state.active_peer else ""
        print(f"  {c(peer_label + '> ', CYAN)}", end="", flush=True)

    async def on_media(peer_id: str, data: bytes, mime: str):
        storage.save_message(peer_id, peer_id, "image_ascii", f"[{mime}]")
        print(c(f"\n  🖼  Media from {peer_id}:", MAGENTA))
        try:
            frames = await render_media_inline(data, mime)
            print(frames[0][0])
        except Exception as e:
            print(c(f"  [render error: {e}]", RED))

    async def on_call_frame(peer_id: str, frame: np.ndarray):
        state.call_frame = frame
        state.in_call = True
        state.call_peer = peer_id

    async def on_call_end(peer_id: str):
        state.in_call = False
        state.call_frame = None
        state.call_peer = None
        print(c(f"\n  Call with {peer_id} ended.", YELLOW))

    async def on_peer_connected(peer_id: str):
        print(c(f"\n  ✓ P2P connection established with {peer_id}", GREEN))

    async def on_error(peer_id: str, msg: str):
        print(c(f"\n  ✗ [{peer_id}]: {msg}", RED))

    # ── Connect to signaling server ───────────────────────────────────────────

    engine = WebRTCEngine(
        signaling_url=SIGNALING_URL,
        user_id=user_id,
        on_text=on_text,
        on_media=on_media,
        on_call_frame=on_call_frame,
        on_call_end=on_call_end,
        on_peer_connected=on_peer_connected,
        on_error=on_error,
    )

    try:
        await engine.connect()
        print(c(f"  ✓ Connected to signaling server", GREEN))
    except Exception as e:
        print(c(f"  ✗ Offline mode ({e})", YELLOW))

    print_banner(user_id, username)

    # Start video render loop in background
    asyncio.create_task(video_render_loop(state))

    # ── Input loop ────────────────────────────────────────────────────────────

    loop = asyncio.get_event_loop()

    try:
        while True:
            peer_label = f"({state.active_peer}) " if state.active_peer else ""
            prompt = f"  {c(peer_label + '> ', CYAN)}"

            # Use asyncio-friendly stdin reading
            sys.stdout.write(prompt)
            sys.stdout.flush()
            raw = await loop.run_in_executor(None, sys.stdin.readline)
            raw = raw.strip()

            if not raw:
                continue

            if raw.startswith("/"):
                keep_running = await handle_command(raw, state, engine, storage)
                if not keep_running:
                    break
            else:
                # Plain message
                if not state.active_peer:
                    print(c("  No active peer. Use /chat <peer_id> first.", RED))
                    continue
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                storage.save_message(state.active_peer, user_id, "text", raw)
                try:
                    await engine.send_text(state.active_peer, raw)
                    print_msg(user_id, raw, user_id)
                except Exception as e:
                    print(c(f"  Send error: {e}", RED))

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print(c("\n  Goodbye!", DIM))
        await engine.close()
        storage.close()


if __name__ == "__main__":
    os.environ.setdefault("TERM", "xterm-256color")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
