#!/usr/bin/env python3
"""
chat_app.py — Full Textual TUI for the P2P Terminal Chat app.

Layout:
  ┌─────────────────────────────────────────────────────────────────┐
  │  🔍 Search / Add user by ID...                        [MY ID]  │
  ├─────────────────┬───────────────────────────────────────────────┤
  │  CONVERSATIONS  │  Chat with: peer_id                          │
  │  ─────────────  │  ───────────────────────────────────────────  │
  │  ● peer_123 📞  │  [10:01] You: hello                          │
  │    peer_456 📞  │  [10:02] peer: hey!                          │
  │                 │                                               │
  ├─────────────────┴───────────────────────────────────────────────┤
  │  Message...                                       [Send] [📞]  │
  └─────────────────────────────────────────────────────────────────┘

Run:
    python chat_app.py
    SIGNALING_URL=wss://your-app.onrender.com python chat_app.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import numpy as np
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    Static,
)
from textual.worker import Worker

from storage import StorageManager
from webrtc_engine import WebRTCEngine
from media_renderer import render_media_inline
from webcam_ascii import frame_to_ascii, terminal_grid, CHAR_ASPECT, RESET


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    """Auto-convert http(s):// to ws(s):// so users don't need to remember."""
    if url.startswith("https://"):
        return "wss://" + url[len("https://"):]
    if url.startswith("http://"):
        return "ws://" + url[len("http://"):]
    return url


def _http_base(ws_url: str) -> str:
    """Convert wss://host to https://host for REST presence checks."""
    if ws_url.startswith("wss://"):
        return "https://" + ws_url[len("wss://"):]
    if ws_url.startswith("ws://"):
        return "http://" + ws_url[len("ws://"):]
    return ws_url


SIGNALING_WS = _normalize_url(os.getenv("SIGNALING_URL", "ws://localhost:8000"))
SIGNALING_HTTP = _http_base(SIGNALING_WS)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))


# ── CSS ───────────────────────────────────────────────────────────────────────

APP_CSS = """
Screen {
    background: #0d1117;
}

/* ── Top search bar ── */
#search-bar {
    height: 3;
    background: #161b22;
    border-bottom: solid #30363d;
    padding: 0 1;
    layout: horizontal;
}

#search-input {
    width: 1fr;
    background: #0d1117;
    border: solid #30363d;
    color: #c9d1d9;
}

#search-input:focus {
    border: solid #58a6ff;
}

#my-id-btn {
    width: auto;
    margin-left: 1;
    background: #21262d;
    color: #58a6ff;
    border: solid #30363d;
    min-width: 16;
}

/* ── Main layout ── */
#main-layout {
    layout: horizontal;
    height: 1fr;
}

/* ── Sidebar ── */
#sidebar {
    width: 28;
    background: #161b22;
    border-right: solid #30363d;
}

#sidebar-title {
    height: 2;
    background: #21262d;
    color: #8b949e;
    text-align: center;
    padding: 0 1;
    border-bottom: solid #30363d;
    text-style: bold;
    content-align: center middle;
}

#conversation-list {
    height: 1fr;
    background: #161b22;
}

ConversationItem {
    height: 4;
    padding: 0 1;
    border-bottom: solid #21262d;
    background: #161b22;
    layout: vertical;
}

ConversationItem:hover {
    background: #1f2937;
}

ConversationItem.active {
    background: #1f2937;
    border-left: thick #58a6ff;
}

.conv-row {
    height: 2;
    layout: horizontal;
    align: left middle;
}

.conv-peer-label {
    width: 1fr;
    color: #c9d1d9;
    text-style: bold;
}

.conv-peer-label.online {
    color: #3fb950;
}

.conv-call-btn {
    width: 4;
    height: 1;
    background: #238636;
    color: #ffffff;
    border: none;
    min-width: 4;
}

.conv-call-btn:hover {
    background: #2ea043;
}

.conv-last-msg {
    color: #8b949e;
    height: 1;
    overflow: hidden;
}

.online-badge {
    color: #3fb950;
    width: 2;
}

.offline-badge {
    color: #6e7681;
    width: 2;
}

/* ── Chat panel ── */
#chat-panel {
    width: 1fr;
    layout: vertical;
}

#chat-header {
    height: 3;
    background: #161b22;
    border-bottom: solid #30363d;
    padding: 0 2;
    content-align: left middle;
    color: #58a6ff;
    text-style: bold;
}

#chat-messages {
    height: 1fr;
    background: #0d1117;
    padding: 1 2;
}

#no-chat-placeholder {
    height: 1fr;
    background: #0d1117;
    content-align: center middle;
    color: #6e7681;
    text-style: italic;
}

.message-block {
    margin-bottom: 1;
}

.msg-mine {
    color: #58a6ff;
}

.msg-theirs {
    color: #3fb950;
}

.msg-time {
    color: #6e7681;
}

.msg-system {
    color: #d29922;
    text-style: italic;
}

/* ── Input bar ── */
#input-bar {
    height: 3;
    background: #161b22;
    border-top: solid #30363d;
    layout: horizontal;
    padding: 0 1;
    align: left middle;
}

#message-input {
    width: 1fr;
    background: #0d1117;
    border: solid #30363d;
    color: #c9d1d9;
}

#message-input:focus {
    border: solid #58a6ff;
}

#send-btn {
    width: 8;
    margin-left: 1;
    background: #1f6feb;
    color: #ffffff;
    border: none;
}

#send-btn:hover {
    background: #388bfd;
}

#call-btn {
    width: 4;
    margin-left: 1;
    background: #238636;
    color: #ffffff;
    border: none;
}

#call-btn:hover {
    background: #2ea043;
}

/* ── Video call screen ── */
#video-screen {
    background: #000000;
    layout: vertical;
}

#video-art {
    height: 1fr;
    background: #000000;
    overflow: hidden;
}

#video-status {
    height: 2;
    background: #161b22;
    color: #f0883e;
    text-style: bold;
    content-align: center middle;
    border-top: solid #30363d;
}

/* ── Status bar ── */
#status-bar {
    height: 1;
    background: #161b22;
    border-top: solid #30363d;
    color: #8b949e;
    padding: 0 2;
    content-align: left middle;
}
"""


# ── Custom Widgets ────────────────────────────────────────────────────────────

class MessageBubble(Static):
    """A single chat message line."""

    def __init__(self, sender_id: str, my_id: str, content: str, timestamp: str, mtype: str = "text"):
        is_mine = sender_id == my_id
        ts = timestamp[:16] if timestamp else ""

        if mtype == "image_ascii":
            text = f"[dim]{ts}[/]  [yellow]{'You' if is_mine else sender_id[:8]}[/] [dim]sent media[/]"
        elif is_mine:
            text = f"[dim]{ts}[/]  [bold blue]You[/]: {content}"
        else:
            text = f"[dim]{ts}[/]  [bold green]{sender_id[:8]}[/]: {content}"

        super().__init__(text, classes="message-block")


class ConversationItem(Widget):
    """A sidebar item representing one conversation."""

    def __init__(self, peer_id: str, last_msg: str = "", online: bool = False):
        # Use a unique id so Textual can track each item in the list
        super().__init__(id=f"conv-item-{peer_id}")
        self.peer_id = peer_id
        self.last_msg = last_msg
        self.is_online = online

    def compose(self) -> ComposeResult:
        badge = "●" if self.is_online else "○"
        badge_class = "online-badge" if self.is_online else "offline-badge"
        label_class = "conv-peer-label online" if self.is_online else "conv-peer-label"

        # Pass children via constructor — avoids NoActiveAppError that occurs
        # when using 'with Horizontal()' context manager on unmounted widgets.
        yield Horizontal(
            Label(badge, classes=badge_class),
            Label(self.peer_id[:14], classes=label_class),
            Button("📞", classes="conv-call-btn", id=f"call-{self.peer_id}"),
            classes="conv-row",
        )
        preview = self.last_msg[:24] + "…" if len(self.last_msg) > 24 else self.last_msg
        yield Label(preview or "No messages yet", classes="conv-last-msg")


# ── Video Call Screen ─────────────────────────────────────────────────────────

class VideoCallScreen(Screen):
    """Full-screen ASCII video call overlay."""

    BINDINGS = [Binding("escape,q", "hangup", "Hang Up")]

    def __init__(self, peer_id: str, engine: WebRTCEngine):
        super().__init__()
        self.peer_id = peer_id
        self.engine = engine
        self._running = True

    def compose(self) -> ComposeResult:
        with Vertical(id="video-screen"):
            yield Static("", id="video-art")
            yield Static(
                f"📞  In call with [bold]{self.peer_id}[/bold]  |  Press ESC or Q to hang up",
                id="video-status"
            )

    def on_mount(self):
        self.set_interval(0.1, self._refresh_frame)

    def _refresh_frame(self):
        app: TerminalChatApp = self.app  # type: ignore
        frame = app.call_frame
        if frame is not None:
            cols = self.size.width
            rows = self.size.height - 2
            h, w = frame.shape[:2]
            fit_rows = max(1, int(cols * CHAR_ASPECT * h / w))
            use_rows = min(rows, fit_rows)
            art = frame_to_ascii(frame, cols, use_rows, color=True, invert=False)
            try:
                self.query_one("#video-art", Static).update(art)
            except NoMatches:
                pass

    async def action_hangup(self):
        self._running = False
        app: TerminalChatApp = self.app  # type: ignore
        await app.engine.end_call(self.peer_id)
        app.call_frame = None
        self.app.pop_screen()


# ── Main App ──────────────────────────────────────────────────────────────────

class TerminalChatApp(App):
    """Main Textual application."""

    CSS = APP_CSS
    TITLE = "Terminal P2P Chat"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
    ]

    # Reactive state
    active_peer: reactive[Optional[str]] = reactive(None)
    status_text: reactive[str] = reactive("Connecting to signaling server…")

    def __init__(self, storage: StorageManager, engine: WebRTCEngine):
        super().__init__()
        self.storage = storage
        self.engine = engine
        self.my_id = storage.get_user_id()
        self.my_name = storage.get_username()
        # peer_id -> list of conversation items to build sidebar
        self.conversations: Dict[str, dict] = {}  # {peer_id: {last_msg, online}}
        # live video frame for the call screen
        self.call_frame: Optional[np.ndarray] = None
        # in-memory message log per peer (in addition to DB)
        self.live_messages: Dict[str, List[dict]] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # ── Top search bar
        with Horizontal(id="search-bar"):
            yield Input(
                placeholder="🔍  Enter a User ID to start chatting…",
                id="search-input"
            )
            yield Button(f"MY ID: {self.my_id}", id="my-id-btn")

        # ── Main layout
        with Horizontal(id="main-layout"):
            # Sidebar
            with Vertical(id="sidebar"):
                yield Static("CONVERSATIONS", id="sidebar-title")
                yield ScrollableContainer(id="conversation-list")

            # Chat panel
            with Vertical(id="chat-panel"):
                yield Static(
                    "Select a conversation or search for a user",
                    id="chat-header"
                )
                yield ScrollableContainer(
                    Static(
                        "💬  No conversation selected.\n\n"
                        "Search for a User ID above or click a conversation on the left.",
                        id="no-chat-placeholder"
                    ),
                    id="chat-messages"
                )

        # ── Input bar
        with Horizontal(id="input-bar"):
            yield Input(placeholder="Type a message…", id="message-input")
            yield Button("Send", id="send-btn")
            yield Button("📞", id="call-btn")

        # ── Status bar
        yield Static(id="status-bar")
        yield Footer()

    def on_mount(self):
        self.query_one("#status-bar", Static).update(
            f"You: {self.my_name} [{self.my_id}]  |  Server: {SIGNALING_WS}"
        )
        # Load existing conversations from DB
        self._load_existing_conversations()
        # Start presence polling
        self.set_interval(10, self._poll_presence)

    def _load_existing_conversations(self):
        """Scan DB for all peer_ids we've talked to and add them to sidebar."""
        try:
            conn = self.storage.conn
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT peer_id FROM messages")
            rows = cursor.fetchall()
            for row in rows:
                peer_id = row["peer_id"]
                history = self.storage.get_chat_history(peer_id, limit=1)
                last = history[-1]["content"] if history else ""
                self._upsert_conversation(peer_id, last_msg=last, online=False)
        except Exception:
            pass

    def _upsert_conversation(self, peer_id: str, last_msg: str = "", online: bool = False):
        """Add or update a conversation in the sidebar."""
        self.conversations[peer_id] = {"last_msg": last_msg, "online": online}
        self._rebuild_sidebar()

    def _rebuild_sidebar(self):
        """Re-render the entire sidebar conversation list."""
        async def _do_rebuild():
            try:
                container = self.query_one("#conversation-list", ScrollableContainer)
                await container.remove_children()
                for peer_id, info in self.conversations.items():
                    item = ConversationItem(
                        peer_id=peer_id,
                        last_msg=info.get("last_msg", ""),
                        online=info.get("online", False),
                    )
                    if peer_id == self.active_peer:
                        item.add_class("active")
                    await container.mount(item)
            except NoMatches:
                pass
        self.call_after_refresh(_do_rebuild)

    @work(exclusive=False, thread=True)
    def _poll_presence(self):
        """Check online status of known peers via the REST endpoint."""
        if not self.conversations:
            return
        try:
            import httpx
            with httpx.Client(timeout=5) as client:
                for peer_id in list(self.conversations.keys()):
                    try:
                        r = client.get(f"{SIGNALING_HTTP}/presence/{peer_id}")
                        if r.status_code == 200:
                            online = r.json().get("online", False)
                            self.conversations[peer_id]["online"] = online
                    except Exception:
                        pass
            self.call_from_thread(self._rebuild_sidebar)
        except Exception:
            pass

    # ── Peer selection ────────────────────────────────────────────────────────

    def watch_active_peer(self, peer_id: Optional[str]):
        """Called when active_peer changes — reload chat history."""
        if peer_id is None:
            return
        self._reload_chat(peer_id)
        try:
            self.query_one("#chat-header", Static).update(
                f"📨  Chat with [bold cyan]{peer_id}[/bold cyan]"
                + ("  🟢 Online" if self.conversations.get(peer_id, {}).get("online") else "  ⚫ Offline")
            )
        except NoMatches:
            pass

    def _reload_chat(self, peer_id: str):
        """Load history from DB + live messages and render into chat panel."""
        history = self.storage.get_chat_history(peer_id, limit=100)
        live = self.live_messages.get(peer_id, [])

        # Merge and deduplicate by timestamp+content (simple approach)
        all_msgs = history + live

        try:
            container = self.query_one("#chat-messages", ScrollableContainer)
            container.remove_children()
            if not all_msgs:
                container.mount(Static(
                    "No messages yet. Say hi! 👋",
                    classes="msg-system"
                ))
            else:
                for msg in all_msgs:
                    container.mount(MessageBubble(
                        sender_id=msg["sender_id"],
                        my_id=self.my_id,
                        content=msg["content"],
                        timestamp=msg.get("timestamp", ""),
                        mtype=msg.get("message_type", "text"),
                    ))
            # Scroll to bottom
            container.scroll_end(animate=False)
        except NoMatches:
            pass

    # ── Event handlers ────────────────────────────────────────────────────────

    @on(Input.Submitted, "#search-input")
    def on_search_submitted(self, event: Input.Submitted):
        peer_id = event.value.strip()
        if not peer_id or peer_id == self.my_id:
            return
        event.input.clear()
        self._upsert_conversation(peer_id)
        self.active_peer = peer_id
        self.query_one("#message-input", Input).focus()

    @on(Button.Pressed, "#my-id-btn")
    def on_myid_pressed(self):
        """Copy user ID to clipboard and show it prominently."""
        try:
            import subprocess
            subprocess.run(["xclip", "-selection", "clipboard"],
                           input=self.my_id.encode(), check=False)
        except Exception:
            pass
        self.notify(f"Your ID: {self.my_id}", title="User ID", severity="information")

    @on(Button.Pressed, "#send-btn")
    async def on_send_pressed(self):
        await self._send_message()

    @on(Input.Submitted, "#message-input")
    async def on_message_submitted(self, event: Input.Submitted):
        await self._send_message()

    async def _send_message(self):
        if not self.active_peer:
            self.notify("Select a conversation first!", severity="warning")
            return
        inp = self.query_one("#message-input", Input)
        text = inp.value.strip()
        if not text:
            return
        inp.clear()

        peer_id = self.active_peer
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save locally
        self.storage.save_message(peer_id, self.my_id, "text", text)
        self.live_messages.setdefault(peer_id, []).append({
            "sender_id": self.my_id,
            "message_type": "text",
            "content": text,
            "timestamp": ts,
        })

        # Update sidebar preview
        if peer_id in self.conversations:
            self.conversations[peer_id]["last_msg"] = text
        self._rebuild_sidebar()

        # Add bubble to chat instantly
        try:
            container = self.query_one("#chat-messages", ScrollableContainer)
            # Remove placeholder if present
            try:
                container.query_one("#no-chat-placeholder").remove()
            except NoMatches:
                pass
            container.mount(MessageBubble(self.my_id, self.my_id, text, ts))
            container.scroll_end(animate=True)
        except NoMatches:
            pass

        # Send via WebRTC
        try:
            await self.engine.send_text(peer_id, text)
        except Exception as e:
            self.notify(f"Send failed: {e}", severity="error")

    @on(Button.Pressed, "#call-btn")
    async def on_call_btn_pressed(self):
        if not self.active_peer:
            self.notify("Select a conversation first!", severity="warning")
            return
        await self._initiate_call(self.active_peer)

    async def on_button_pressed(self, event: Button.Pressed):
        """Handle sidebar 📞 buttons (id pattern: call-<peer_id>)."""
        btn_id = event.button.id or ""
        if btn_id.startswith("call-"):
            peer_id = btn_id[len("call-"):]
            self.active_peer = peer_id
            await self._initiate_call(peer_id)

    async def _initiate_call(self, peer_id: str):
        """Check presence, then start a call."""
        online = self.conversations.get(peer_id, {}).get("online", False)
        if not online:
            self.notify(
                f"{peer_id} appears offline. Trying anyway…",
                severity="warning"
            )
        self.notify(f"Calling {peer_id}…", severity="information")
        try:
            await self.engine.start_call(peer_id, CAMERA_INDEX)
            await self.push_screen(VideoCallScreen(peer_id, self.engine))
        except Exception as e:
            self.notify(f"Call failed: {e}", severity="error")

    # ── WebRTC event callbacks (called from engine) ───────────────────────────

    async def on_incoming_text(self, peer_id: str, text: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.storage.save_message(peer_id, peer_id, "text", text)
        self.live_messages.setdefault(peer_id, []).append({
            "sender_id": peer_id,
            "message_type": "text",
            "content": text,
            "timestamp": ts,
        })
        self._upsert_conversation(peer_id, last_msg=text)

        if self.active_peer == peer_id:
            try:
                container = self.query_one("#chat-messages", ScrollableContainer)
                container.mount(MessageBubble(peer_id, self.my_id, text, ts))
                container.scroll_end(animate=True)
            except NoMatches:
                pass
        else:
            self.notify(f"New message from {peer_id}", title="💬 Message")

    async def on_incoming_media(self, peer_id: str, data: bytes, mime: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.storage.save_message(peer_id, peer_id, "image_ascii", f"[{mime}]")
        self._upsert_conversation(peer_id, last_msg=f"[sent media]")

        frames = await render_media_inline(data, mime)
        ascii_art = frames[0][0]

        self.live_messages.setdefault(peer_id, []).append({
            "sender_id": peer_id,
            "message_type": "image_ascii",
            "content": ascii_art,
            "timestamp": ts,
        })
        if self.active_peer == peer_id:
            try:
                container = self.query_one("#chat-messages", ScrollableContainer)
                container.mount(MessageBubble(peer_id, self.my_id, ascii_art, ts, "image_ascii"))
                container.scroll_end(animate=True)
            except NoMatches:
                pass
        else:
            self.notify(f"Media from {peer_id}", title="🖼️ Image")

    async def on_incoming_call_frame(self, peer_id: str, frame: np.ndarray):
        self.call_frame = frame
        # If call screen isn't open yet, push it
        if not any(isinstance(s, VideoCallScreen) for s in self.screen_stack):
            await self.push_screen(VideoCallScreen(peer_id, self.engine))

    async def on_call_ended(self, peer_id: str):
        self.call_frame = None
        self.notify(f"Call with {peer_id} ended", severity="warning")
        # Pop video screen if open
        if any(isinstance(s, VideoCallScreen) for s in self.screen_stack):
            self.pop_screen()

    async def on_peer_connected(self, peer_id: str):
        self.conversations.setdefault(peer_id, {})["online"] = True
        self._rebuild_sidebar()
        self.notify(f"✓ Connected to {peer_id}", severity="information")

    async def on_error(self, peer_id: str, msg: str):
        self.notify(f"Error [{peer_id}]: {msg}", severity="error")


# ── Entry point ───────────────────────────────────────────────────────────────

async def run():
    storage = StorageManager()

    # First-run setup via simple terminal prompt (before TUI launches)
    username = storage.get_username()
    if username.startswith("User_"):
        print(f"\n  Welcome to Terminal P2P Chat!")
        print(f"  Your auto-generated User ID: \033[96m{storage.get_user_id()}\033[0m")
        new_name = input("  Choose a username: ").strip()
        if new_name:
            storage.update_username(new_name)

    # Wire up WebRTC engine callbacks to the app
    # We use a placeholder app reference that gets filled in after construction
    app_ref: list[TerminalChatApp] = []

    async def on_text(peer_id: str, text: str):
        if app_ref:
            await app_ref[0].on_incoming_text(peer_id, text)

    async def on_media(peer_id: str, data: bytes, mime: str):
        if app_ref:
            await app_ref[0].on_incoming_media(peer_id, data, mime)

    async def on_call_frame(peer_id: str, frame):
        if app_ref:
            await app_ref[0].on_incoming_call_frame(peer_id, frame)

    async def on_call_end(peer_id: str):
        if app_ref:
            await app_ref[0].on_call_ended(peer_id)

    async def on_peer_connected(peer_id: str):
        if app_ref:
            await app_ref[0].on_peer_connected(peer_id)

    async def on_error(peer_id: str, msg: str):
        if app_ref:
            await app_ref[0].on_error(peer_id, msg)

    engine = WebRTCEngine(
        signaling_url=SIGNALING_WS,
        user_id=storage.get_user_id(),
        on_text=on_text,
        on_media=on_media,
        on_call_frame=on_call_frame,
        on_call_end=on_call_end,
        on_peer_connected=on_peer_connected,
        on_error=on_error,
    )

    # Try connecting (non-fatal if offline)
    try:
        await engine.connect()
        print(f"  ✓ Connected to signaling server")
    except Exception as e:
        print(f"  ✗ Offline mode: {e}")

    app = TerminalChatApp(storage=storage, engine=engine)
    app_ref.append(app)

    try:
        await app.run_async()
    finally:
        await engine.close()
        storage.close()


if __name__ == "__main__":
    os.environ.setdefault("TERM", "xterm-256color")
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
