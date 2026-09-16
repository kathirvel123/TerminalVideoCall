#!/usr/bin/env python3
"""
chat_app.py — Main TUI entry point for the P2P Terminal Chat app.

Run:
    python chat_app.py

First run: prompts you to set a username. Your unique ID is auto-generated
and stored in profile.json.

Slash commands:
    /call <peer_id>        — start a live ASCII video call
    /hangup                — end the current call
    /send <filepath>       — send an image or GIF to the current peer
    /chat <peer_id>        — switch active chat to a different peer
    /name <username>       — change your display name
    /myid                  — print your unique user ID
    /quit                  — exit the app

All chat history is stored locally in chat_history.db.
No chat data ever touches the server.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

from storage import StorageManager
from webrtc_engine import WebRTCEngine
from media_renderer import render_media_inline
from webcam_ascii import frame_to_ascii, terminal_grid, CHAR_ASPECT, HOME, CLEAR, RESET, HIDE_CURSOR, SHOW_CURSOR

# ── configuration ────────────────────────────────────────────────────────────

SIGNALING_URL = os.getenv("SIGNALING_URL", "ws://localhost:8000")
CAMERA_INDEX  = int(os.getenv("CAMERA_INDEX", "0"))

# ── ANSI helpers ─────────────────────────────────────────────────────────────

BOLD     = "\033[1m"
DIM      = "\033[2m"
CYAN     = "\033[36m"
GREEN    = "\033[32m"
YELLOW   = "\033[33m"
RED      = "\033[31m"
MAGENTA  = "\033[35m"
BLUE     = "\033[34m"

def color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"

def divider(cols: int = 60, char: str = "─") -> str:
    return color(char * cols, DIM)


# ── chat state ────────────────────────────────────────────────────────────────

class ChatState:
    def __init__(self):
        self.active_peer: Optional[str] = None
        self.in_call: bool = False
        self.call_peer: Optional[str] = None
        # latest video frame from current call (BGR numpy array)
        self.call_frame: Optional[np.ndarray] = None
        # accumulated inline ASCII media lines per peer
        self.media_display: Dict[str, List[str]] = {}


# ── display helpers ──────────────────────────────────────────────────────────

def print_banner(user_id: str, username: str):
    cols, _ = terminal_grid(None)
    banner = f"""
{color('╔' + '═' * (cols - 2) + '╗', CYAN)}
{color('║', CYAN)}  {color('⬡  TERMINAL P2P CHAT', BOLD + CYAN)}  {color(f'you: {username}  [{user_id}]', DIM)}
{color('╚' + '═' * (cols - 2) + '╝', CYAN)}
{color('Type /myid to share your ID  |  /call <peer_id>  |  /help for all commands', DIM)}
"""
    print(banner)


def format_message(msg: dict, my_id: str, peer_id: str) -> str:
    sender = msg["sender_id"]
    ts = msg.get("timestamp", "")
    mtype = msg.get("message_type", "text")
    content = msg["content"]

    ts_str = color(f"[{ts[:16]}]", DIM) if ts else ""

    if mtype == "image_ascii":
        # Stored as a placeholder; actual render happens live on receive
        return f"{ts_str}  {color(sender, YELLOW)} sent an image:\n{content}"

    if sender == my_id:
        label = color("You", GREEN + BOLD)
    else:
        label = color(sender[:8], MAGENTA + BOLD)

    return f"{ts_str}  {label}: {content}"


def print_chat_history(
    history: List[dict],
    my_id: str,
    peer_id: str,
    cols: int,
    rows: int,
    media_lines: List[str],
):
    sys.stdout.write(CLEAR + HOME)
    # Header
    print(divider(cols))
    print(color(f"  Chat with {peer_id}", CYAN + BOLD))
    print(divider(cols))

    # Messages (last N that fit)
    lines_budget = rows - 8
    formatted = [format_message(m, my_id, peer_id) for m in history]
    formatted += media_lines
    display = formatted[-lines_budget:]
    for line in display:
        print(line)

    print(divider(cols))
    sys.stdout.flush()


def print_call_frame(frame: np.ndarray, peer_id: str, cols: int, rows: int):
    """Render a live video frame as ASCII and write it to stdout."""
    h, w = frame.shape[:2]
    fit_rows = max(1, int(cols * CHAR_ASPECT * h / w))
    use_rows = min(rows - 2, fit_rows)
    art = frame_to_ascii(frame, cols, use_rows, color=True, invert=False)
    status = color(f"  📞 In call with {peer_id}  |  /hangup to end", YELLOW)
    sys.stdout.write(HOME + art + "\n" + status[:cols].ljust(cols))
    sys.stdout.flush()


# ── command handlers ─────────────────────────────────────────────────────────

async def handle_command(
    raw: str,
    state: ChatState,
    engine: WebRTCEngine,
    storage: StorageManager,
) -> Optional[str]:
    """
    Process a slash command.
    Returns a status message to print, or None.
    """
    parts = raw.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        await engine.close()
        storage.close()
        sys.stdout.write(SHOW_CURSOR + RESET + "\n")
        sys.exit(0)

    elif cmd == "/myid":
        return color(f"Your User ID: {storage.get_user_id()}", CYAN)

    elif cmd == "/help":
        return (
            f"\n{color('Commands:', BOLD + CYAN)}\n"
            f"  /chat <peer_id>   — open chat with a peer\n"
            f"  /call <peer_id>   — start ASCII video call\n"
            f"  /hangup           — end current call\n"
            f"  /send <filepath>  — send image or GIF\n"
            f"  /name <username>  — change your username\n"
            f"  /myid             — show your User ID\n"
            f"  /quit             — exit\n"
        )

    elif cmd == "/name":
        if not arg:
            return color("Usage: /name <new_username>", RED)
        storage.update_username(arg)
        return color(f"Username updated to: {arg}", GREEN)

    elif cmd == "/chat":
        if not arg:
            return color("Usage: /chat <peer_id>", RED)
        state.active_peer = arg
        return color(f"Switched to chat with {arg}", CYAN)

    elif cmd == "/call":
        peer = arg or state.active_peer
        if not peer:
            return color("Usage: /call <peer_id>", RED)
        state.in_call = True
        state.call_peer = peer
        await engine.start_call(peer, CAMERA_INDEX)
        return color(f"Calling {peer}...", YELLOW)

    elif cmd == "/hangup":
        if state.call_peer:
            await engine.end_call(state.call_peer)
            state.in_call = False
            state.call_frame = None
            state.call_peer = None
        return color("Call ended.", RED)

    elif cmd == "/send":
        if not arg:
            return color("Usage: /send <filepath>", RED)
        if not state.active_peer:
            return color("No active peer. Use /chat <peer_id> first.", RED)
        if not os.path.exists(arg):
            return color(f"File not found: {arg}", RED)
        with open(arg, "rb") as f:
            data = f.read()
        ext = arg.rsplit(".", 1)[-1].lower()
        mime = "image/gif" if ext == "gif" else f"image/{ext}"
        await engine.send_media(state.active_peer, data, mime)

        # Also render it locally and store as a placeholder
        frames = await render_media_inline(data, mime)
        ascii_art = frames[0][0]  # first frame for static display
        state.media_display.setdefault(state.active_peer, []).append(
            color(f"[You sent {os.path.basename(arg)}]", GREEN) + "\n" + ascii_art
        )
        storage.save_message(state.active_peer, storage.get_user_id(), "image_ascii", f"[sent {arg}]")
        return color(f"Sent: {arg}", GREEN)

    else:
        return color(f"Unknown command: {cmd}. Type /help for a list.", RED)


# ── main app loop ─────────────────────────────────────────────────────────────

async def main():
    storage = StorageManager()

    # First-run setup
    username = storage.get_username()
    user_id  = storage.get_user_id()
    if username.startswith("User_"):
        print(color("First run! Let's set up your profile.", CYAN + BOLD))
        new_name = input("Enter a username: ").strip()
        if new_name:
            storage.update_username(new_name)
            username = new_name

    state = ChatState()

    # ── WebRTC callbacks ──────────────────────────────────────────────────

    async def on_text(peer_id: str, text: str):
        storage.save_message(peer_id, peer_id, "text", text)
        if state.active_peer == peer_id:
            # Will be shown on next history refresh
            pass
        else:
            print(color(f"\n[New message from {peer_id}]", YELLOW))

    async def on_media(peer_id: str, data: bytes, mime: str):
        frames = await render_media_inline(data, mime)
        ascii_art = frames[0][0]
        state.media_display.setdefault(peer_id, []).append(
            color(f"[{peer_id} sent {'a GIF' if mime == 'image/gif' else 'an image'}]", MAGENTA) + "\n" + ascii_art
        )
        storage.save_message(peer_id, peer_id, "image_ascii", f"[received {mime}]")
        if state.active_peer != peer_id:
            print(color(f"\n[Media received from {peer_id}]", YELLOW))

    async def on_call_frame(peer_id: str, frame: np.ndarray):
        state.call_frame = frame
        state.in_call = True
        state.call_peer = peer_id

    async def on_call_end(peer_id: str):
        state.in_call = False
        state.call_frame = None
        state.call_peer = None
        print(color(f"\nCall with {peer_id} ended.", RED))

    async def on_peer_connected(peer_id: str):
        print(color(f"\n✓ Connected to {peer_id}", GREEN))

    async def on_error(peer_id: str, msg: str):
        print(color(f"\n✗ Error [{peer_id}]: {msg}", RED))

    # ── connect engine ────────────────────────────────────────────────────

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
    except Exception as e:
        print(color(f"Could not connect to signaling server: {e}", RED))
        print(color(f"Set SIGNALING_URL env var to point to your deployed server.", DIM))
        print(color(f"Continuing in offline mode (no P2P connections).", YELLOW))

    # ── TUI loop ─────────────────────────────────────────────────────────

    sys.stdout.write(HIDE_CURSOR)
    print_banner(user_id, username)

    prompt_style = Style.from_dict({
        "prompt":  "ansicyan bold",
        "bottom-toolbar": "bg:ansiblue fg:ansiwhite",
    })
    session = PromptSession(style=prompt_style)

    async def refresh_display():
        """Background task: refresh the chat or call view at ~10 fps."""
        cols, rows = terminal_grid(None)
        while True:
            cols, rows = terminal_grid(None)
            if state.in_call and state.call_frame is not None:
                print_call_frame(state.call_frame, state.call_peer or "peer", cols, rows)
            elif state.active_peer:
                history = storage.get_chat_history(state.active_peer, limit=rows)
                media = state.media_display.get(state.active_peer, [])
                print_chat_history(history, user_id, state.active_peer, cols, rows, media)
            await asyncio.sleep(0.1)

    asyncio.create_task(refresh_display())

    with patch_stdout():
        while True:
            try:
                peer_label = f" ({state.active_peer})" if state.active_peer else ""
                prompt_text = HTML(f"<ansicyan><b>you{peer_label} &gt; </b></ansicyan>")
                raw = await session.prompt_async(prompt_text)
                raw = raw.strip()
                if not raw:
                    continue

                if raw.startswith("/"):
                    result = await handle_command(raw, state, engine, storage)
                    if result:
                        print(result)
                else:
                    # Plain text message
                    if not state.active_peer:
                        print(color("No active peer. Use /chat <peer_id> first.", RED))
                        continue
                    storage.save_message(state.active_peer, user_id, "text", raw)
                    try:
                        await engine.send_text(state.active_peer, raw)
                    except Exception as e:
                        print(color(f"Send error: {e}", RED))

            except (KeyboardInterrupt, EOFError):
                break

    await engine.close()
    storage.close()
    sys.stdout.write(SHOW_CURSOR + RESET + "\n")


if __name__ == "__main__":
    os.environ.setdefault("TERM", "xterm-256color")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
