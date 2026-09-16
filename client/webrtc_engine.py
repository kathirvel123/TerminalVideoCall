#!/usr/bin/env python3
"""
webrtc_engine.py — P2P WebRTC engine for Terminal Chat.

Responsibilities:
  - Connect to the signaling server via WebSocket using the local user ID.
  - For each peer, create an RTCPeerConnection with:
      * 'chat'  DataChannel  → text messages (JSON)
      * 'media' DataChannel  → binary image/GIF transfer
  - Attach a webcam VideoStreamTrack when a /call is initiated.
  - Drive SDP offer/answer and ICE-candidate exchange through the
    signaling server (the server never sees media content).
  - Fire callbacks so the TUI layer can react to events without
    caring about WebRTC internals.

Usage:
    engine = WebRTCEngine(
        signaling_url="wss://your-server.onrender.com",
        user_id="abc12345",
        on_text=...,
        on_media=...,
        on_call_start=...,
        on_call_end=...,
    )
    await engine.connect()
    await engine.send_text("peer_id", "Hello!")
    await engine.send_media("peer_id", b"<raw image bytes>", "image/jpeg")
    await engine.start_call("peer_id")
    await engine.close()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from typing import Callable, Awaitable, Dict, Optional, Any

import cv2
import numpy as np
import websockets
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceCandidate,
    MediaStreamTrack,
)
from aiortc.contrib.media import MediaBlackhole, MediaPlayer
from av import VideoFrame

logger = logging.getLogger(__name__)

# Public Google STUN servers — no setup required.
# For calls across restrictive NATs, set TURN_URL/TURN_USERNAME/TURN_PASSWORD env vars.
ICE_SERVERS = [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]
_turn = os.getenv("TURN_URL")
if _turn:
    _turn_cfg = {"urls": [_turn]}
    if os.getenv("TURN_USERNAME"):
        _turn_cfg["username"] = os.getenv("TURN_USERNAME")
        _turn_cfg["credential"] = os.getenv("TURN_PASSWORD", "")
    ICE_SERVERS.append(_turn_cfg)


# ---------------------------------------------------------------------------
# Webcam video track for live calls
# ---------------------------------------------------------------------------

class WebcamVideoTrack(MediaStreamTrack):
    """Reads frames from OpenCV and feeds them into the WebRTC pipeline."""

    kind = "video"

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 15)

    async def recv(self) -> VideoFrame:
        ok, bgr = self._cap.read()
        if not ok:
            # Return a black frame if camera fails
            bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        bgr = cv2.flip(bgr, 1)  # mirror selfie
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts, frame.time_base = self._pts, self._time_base
        return frame

    @property
    def _pts(self):
        import time
        return int(time.perf_counter() * 1_000_000)

    @property
    def _time_base(self):
        from fractions import Fraction
        return Fraction(1, 1_000_000)

    def stop(self):
        super().stop()
        self._cap.release()


# ---------------------------------------------------------------------------
# Per-peer connection wrapper
# ---------------------------------------------------------------------------

class PeerSession:
    """Manages the RTCPeerConnection and DataChannels for a single remote peer."""

    def __init__(
        self,
        peer_id: str,
        pc: RTCPeerConnection,
        on_text: Callable[[str, str], Awaitable[None]],
        on_media: Callable[[str, bytes, str], Awaitable[None]],
        on_call_frame: Callable[[str, np.ndarray], Awaitable[None]],
    ):
        self.peer_id = peer_id
        self.pc = pc
        self._on_text = on_text
        self._on_media = on_media
        self._on_call_frame = on_call_frame
        self._chat_channel = None
        self._media_channel = None
        self._webcam_track: Optional[WebcamVideoTrack] = None
        self._media_buf: Dict[int, bytes] = {}  # chunk_idx -> bytes
        self._media_total: int = 0
        self._media_mime: str = ""
        self._pending_ice: list[RTCIceCandidate] = []
        self._ice_waiters: list[asyncio.Future] = []
        self.connected = False
        self._pending_chat: list[str] = []
        self._pending_media: list[tuple[bytes, str]] = []
        self._ready_event: Optional[asyncio.Event] = None

    def _flush_pending_ice(self):
        pending, self._pending_ice = self._pending_ice, []
        for fut in self._ice_waiters:
            if not fut.done():
                fut.set_result(None)
        self._ice_waiters = []
        return pending

    async def add_ice(self, candidate: RTCIceCandidate):
        """Add an ICE candidate, queueing it if the remote description isn't set yet."""
        if self.pc.remoteDescription is None:
            self._pending_ice.append(candidate)
        else:
            try:
                await self.pc.addIceCandidate(candidate)
            except Exception as e:
                logger.warning(f"addIceCandidate for {self.peer_id} failed: {e}")

    def _flush_pending_chat(self):
        if not self._chat_channel or self._chat_channel.readyState != "open":
            return
        while self._pending_chat:
            text = self._pending_chat.pop(0)
            try:
                self._chat_channel.send(json.dumps({"text": text}))
            except Exception as e:
                logger.warning(f"Failed to flush chat for {self.peer_id}: {e}")
                self._pending_chat.insert(0, text)
                break

    def _flush_pending_media(self):
        if not self._media_channel or self._media_channel.readyState != "open":
            return
        while self._pending_media:
            data, mime = self._pending_media.pop(0)
            try:
                self._media_channel.send(json.dumps({"mime": mime, "total_chunks": 1}))
                self._media_channel.send(data)
            except Exception as e:
                logger.warning(f"Failed to flush media for {self.peer_id}: {e}")
                self._pending_media.insert(0, (data, mime))
                break

    def attach_chat_channel(self, channel):
        self._chat_channel = channel
        channel.on("message")(self._handle_chat_message)
        channel.on("open")(self._on_channel_open)

    def attach_media_channel(self, channel):
        self._media_channel = channel
        channel.on("message")(self._handle_media_message)
        channel.on("open")(self._on_channel_open)

    async def _on_channel_open(self):
        self.connected = True
        if self._ready_event and not self._ready_event.is_set():
            self._ready_event.set()
        self._flush_pending_chat()
        self._flush_pending_media()

    async def _handle_chat_message(self, message: str):
        try:
            msg = json.loads(message)
            await self._on_text(self.peer_id, msg.get("text", ""))
        except Exception as e:
            logger.warning(f"Bad chat message from {self.peer_id}: {e}")

    async def _handle_media_message(self, data):
        """
        Media is sent in chunks:
          Header (first message): JSON {"mime": "image/jpeg", "total_chunks": N}
          Chunks: binary bytes (in order, index implicit by arrival)
        """
        if isinstance(data, str):
            try:
                header = json.loads(data)
                self._media_mime = header.get("mime", "image/jpeg")
                self._media_total = int(header.get("total_chunks", 1))
                self._media_buf = {}
            except Exception as e:
                logger.warning(f"Bad media header from {self.peer_id}: {e}")
        else:
            idx = len(self._media_buf)
            self._media_buf[idx] = data
            if len(self._media_buf) == self._media_total:
                payload = b"".join(self._media_buf[i] for i in range(self._media_total))
                await self._on_media(self.peer_id, payload, self._media_mime)
                self._media_buf = {}

    async def send_text(self, text: str) -> bool:
        if not self._chat_channel:
            logger.warning(f"Chat channel not created yet for {self.peer_id}")
            return False
        state = self._chat_channel.readyState
        if state == "open":
            self._chat_channel.send(json.dumps({"text": text}))
            return True
        if state == "connecting":
            # Buffer the message — it will be flushed when the channel opens
            self._pending_chat.append(text)
            return True
        logger.warning(f"Chat channel closed for {self.peer_id}")
        return False

    async def send_media(self, data: bytes, mime: str, chunk_size: int = 16384):
        """Chunk and send binary media over the media DataChannel."""
        state = (
            self._media_channel.readyState if self._media_channel else "closed"
        )
        if state == "open":
            chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
            self._media_channel.send(json.dumps({"mime": mime, "total_chunks": len(chunks)}))
            for chunk in chunks:
                self._media_channel.send(chunk)
        elif state == "connecting":
            self._pending_media.append((data, mime))
        else:
            logger.warning(f"Media channel not open for {self.peer_id}")

    def start_webcam(self, camera_index: int = 0):
        self._webcam_track = WebcamVideoTrack(camera_index)
        self.pc.addTrack(self._webcam_track)

    async def handle_incoming_track(self, track):
        """Decode incoming video frames and fire on_call_frame callback."""
        while True:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                await self._on_call_frame(self.peer_id, img)
            except Exception:
                break

    async def close(self):
        if self._webcam_track:
            self._webcam_track.stop()
        await self.pc.close()


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class WebRTCEngine:
    """
    Top-level engine. One instance per running client app.

    Callbacks (all async):
        on_text(peer_id, text)               — incoming chat message
        on_media(peer_id, data, mime)        — incoming image/GIF bytes
        on_call_frame(peer_id, bgr_ndarray)  — incoming video frame during a call
        on_call_end(peer_id)                 — remote peer ended the call
        on_peer_connected(peer_id)           — DataChannels are open and ready
        on_error(peer_id, error_message)     — something went wrong
    """

    def __init__(
        self,
        signaling_url: str,
        user_id: str,
        on_text: Callable = None,
        on_media: Callable = None,
        on_call_frame: Callable = None,
        on_call_end: Callable = None,
        on_peer_connected: Callable = None,
        on_error: Callable = None,
        camera_index: int = 0,
    ):
        # Auto-convert http(s):// → ws(s):// so users can paste their Render URL directly
        if signaling_url.startswith("https://"):
            signaling_url = "wss://" + signaling_url[len("https://"):]
        elif signaling_url.startswith("http://"):
            signaling_url = "ws://" + signaling_url[len("http://"):]

        self._url = signaling_url
        self._user_id = user_id
        self._ws: Optional[Any] = None
        self._peers: Dict[str, PeerSession] = {}
        self._running = False
        self._camera_index = camera_index

        # Default no-op callbacks
        async def noop(*_): pass
        self._on_text = on_text or noop
        self._on_media = on_media or noop
        self._on_call_frame = on_call_frame or noop
        self._on_call_end = on_call_end or noop
        self._on_peer_connected = on_peer_connected or noop
        self._on_error = on_error or noop

    async def connect(self):
        """Connect to the signaling server and start listening for messages."""
        ws_url = f"{self._url}/ws/{self._user_id}"
        self._ws = await websockets.connect(ws_url)
        self._running = True
        logger.info(f"Connected to signaling server as {self._user_id}")
        asyncio.create_task(self._recv_loop())

    async def _recv_loop(self):
        """Background task: receive and dispatch signaling messages."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                    await self._dispatch(msg)
                except Exception as e:
                    logger.error(f"Error dispatching message: {e}")
                    traceback.print_exc()
        except websockets.ConnectionClosed:
            logger.info("Signaling WebSocket closed")
            self._running = False

    async def _dispatch(self, msg: dict):
        sender = msg.get("sender_id")
        mtype = msg.get("type")
        payload = msg.get("payload", {})

        if mtype == "error":
            await self._on_error(sender, payload.get("message", "Unknown error"))
            return

        if mtype == "call_request":
            # Remote peer wants to call us — create a callee session
            session = await self._get_or_create_session(sender, is_offerer=False)
            # Add our webcam so the answer includes video media
            if not session._webcam_track:
                session.start_webcam(self._camera_index)
            logger.info(f"Incoming call from {sender}")

        elif mtype == "offer":
            session = await self._get_or_create_session(sender, is_offerer=False)
            # Ensure our webcam is attached so the answer includes video media
            if not session._webcam_track:
                session.start_webcam(self._camera_index)
            await session.pc.setRemoteDescription(
                RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])
            )
            # Flush any ICE candidates that arrived before the remote description
            for candidate in session._flush_pending_ice():
                await session.add_ice(candidate)
            answer = await session.pc.createAnswer()
            await session.pc.setLocalDescription(answer)
            await self._send_signal(sender, "answer", {
                "sdp": session.pc.localDescription.sdp,
                "type": session.pc.localDescription.type,
            })

        elif mtype == "answer":
            session = self._peers.get(sender)
            if session:
                await session.pc.setRemoteDescription(
                    RTCSessionDescription(sdp=payload["sdp"], type=payload["type"])
                )
                # Flush any ICE candidates that arrived before the remote description
                for candidate in session._flush_pending_ice():
                    await session.add_ice(candidate)

        elif mtype == "ice":
            session = self._peers.get(sender)
            if session and payload:
                candidate = RTCIceCandidate(
                    component=payload.get("component", 1),
                    foundation=payload.get("foundation", ""),
                    ip=payload.get("ip", ""),
                    port=payload.get("port", 0),
                    priority=payload.get("priority", 0),
                    protocol=payload.get("protocol", "udp"),
                    type=payload.get("type", "host"),
                    sdpMid=payload.get("sdpMid"),
                    sdpMLineIndex=payload.get("sdpMLineIndex"),
                )
                await session.add_ice(candidate)

    async def _get_or_create_session(self, peer_id: str, is_offerer: bool) -> PeerSession:
        if peer_id in self._peers:
            return self._peers[peer_id]

        pc = RTCPeerConnection()
        session = PeerSession(
            peer_id=peer_id,
            pc=pc,
            on_text=self._on_text,
            on_media=self._on_media,
            on_call_frame=self._on_call_frame,
        )
        self._peers[peer_id] = session

        # ICE candidate handler
        @pc.on("icecandidate")
        async def on_ice(candidate):
            if candidate:
                await self._send_signal(peer_id, "ice", {
                    "component": candidate.component,
                    "foundation": candidate.foundation,
                    "ip": candidate.ip,
                    "port": candidate.port,
                    "priority": candidate.priority,
                    "protocol": candidate.protocol,
                    "type": candidate.type,
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                })

        # Log connection state changes
        @pc.on("connectionstatechange")
        def on_conn_state():
            logger.info(f"Connection state [{peer_id}]: {pc.connectionState}")
            if pc.connectionState == "failed":
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(
                    asyncio.ensure_future,
                    self._on_error(peer_id, "P2P connection failed. Both users may be behind restrictive NATs — set TURN_URL env var if needed."),
                )

        @pc.on("iceconnectionstatechange")
        def on_ice_state():
            logger.info(f"ICE state [{peer_id}]: {pc.iceConnectionState}")

        # Incoming track (remote video)
        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                asyncio.create_task(session.handle_incoming_track(track))

        if is_offerer:
            # Create both DataChannels as offerer
            chat_ch = pc.createDataChannel("chat")
            media_ch = pc.createDataChannel("media")
            session.attach_chat_channel(chat_ch)
            session.attach_media_channel(media_ch)

            @chat_ch.on("open")
            async def on_open():
                await self._on_peer_connected(peer_id)
        else:
            # Accept incoming DataChannels as answerer
            @pc.on("datachannel")
            def on_datachannel(channel):
                if channel.label == "chat":
                    session.attach_chat_channel(channel)

                    @channel.on("open")
                    async def on_open():
                        await self._on_peer_connected(peer_id)
                elif channel.label == "media":
                    session.attach_media_channel(channel)

        return session

    async def _send_signal(self, target_id: str, mtype: str, payload: dict):
        msg = json.dumps({
            "target_id": target_id,
            "type": mtype,
            "payload": payload,
        })
        if self._ws:
            await self._ws.send(msg)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def send_text(self, peer_id: str, text: str):
        """Open a connection to peer_id if needed and send a text message."""
        session = await self._initiate_if_needed(peer_id)
        ok = await session.send_text(text)
        if not ok:
            raise ConnectionError(
                f"DataChannel not ready for {peer_id}. "
                "Try again once both peers are online and connected."
            )

    async def send_media(self, peer_id: str, data: bytes, mime: str):
        """Send raw image or GIF bytes to a peer."""
        session = await self._initiate_if_needed(peer_id)
        await session.send_media(data, mime)

    async def start_call(self, peer_id: str, camera_index: int = 0):
        """Initiate a video call with a peer."""
        await self._send_signal(peer_id, "call_request", {})
        session = await self._initiate_if_needed(peer_id, with_webcam=True, camera_index=camera_index)

    async def end_call(self, peer_id: str):
        session = self._peers.get(peer_id)
        if session and session._webcam_track:
            session._webcam_track.stop()
            session._webcam_track = None
        await self._on_call_end(peer_id)

    async def _initiate_if_needed(self, peer_id: str, with_webcam: bool = False, camera_index: int = 0) -> PeerSession:
        """Create a session and send an offer if no session exists yet."""
        if peer_id not in self._peers:
            session = await self._get_or_create_session(peer_id, is_offerer=True)
            if with_webcam:
                session.start_webcam(camera_index)
            offer = await session.pc.createOffer()
            await session.pc.setLocalDescription(offer)
            await self._send_signal(peer_id, "offer", {
                "sdp": session.pc.localDescription.sdp,
                "type": session.pc.localDescription.type,
            })
        else:
            session = self._peers[peer_id]
            if with_webcam and not session._webcam_track:
                # Existing chat session — add webcam and renegotiate
                session.start_webcam(camera_index)
                offer = await session.pc.createOffer()
                await session.pc.setLocalDescription(offer)
                await self._send_signal(peer_id, "offer", {
                    "sdp": session.pc.localDescription.sdp,
                    "type": session.pc.localDescription.type,
                })
        return session

    async def close(self):
        self._running = False
        for session in self._peers.values():
            await session.close()
        self._peers.clear()
        if self._ws:
            await self._ws.close()
