# Terminal P2P Chat — Client

A fully peer-to-peer terminal chat app with live ASCII video calls, image/GIF sharing, and local-only chat history.

## Features
- 💬 **Text chat** — messages flow P2P via WebRTC DataChannels
- 🖼️ **Image/GIF sharing** — any image is sent P2P and rendered as ASCII art inline
- 📞 **Live ASCII video calls** — webcam streamed P2P and rendered as colored ASCII in your terminal
- 🔒 **Local-only storage** — all chat history saved in `chat_history.db` on your machine only
- 🌐 **Works across the internet** — no static IP needed (STUN-based NAT traversal)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Point to your deployed signaling server
export SIGNALING_URL=wss://your-app.onrender.com
python chat_app.py

# Or run against a local server for testing
export SIGNALING_URL=ws://localhost:8000
python chat_app.py
```

On first run, you will be prompted to set a username.
Your unique **User ID** is auto-generated and saved in `profile.json`.

## Slash Commands

| Command | Description |
|---|---|
| `/myid` | Print your User ID to share with others |
| `/chat <peer_id>` | Open chat with a peer |
| `/call <peer_id>` | Start an ASCII video call |
| `/hangup` | End the current video call |
| `/send <filepath>` | Send an image or GIF to the active peer |
| `/name <username>` | Change your display name |
| `/help` | Show all commands |
| `/quit` | Exit the app |

## How it works

```
You ──────── WebSocket ──────── Signaling Server ──────── WebSocket ──────── Peer
              (offer/answer only, no media)

You ══════════════════════ WebRTC P2P ══════════════════════ Peer
              (all chat, images, and video go here)
```

The signaling server only touches tiny handshake messages (~1 KB total).
After that, everything is direct peer-to-peer.

## Files

| File | Purpose |
|---|---|
| `chat_app.py` | Main TUI entry point |
| `webrtc_engine.py` | WebRTC P2P connection engine |
| `storage.py` | Local profile (JSON) + chat history (SQLite) |
| `media_renderer.py` | Image/GIF → ASCII art converter |
| `webcam_ascii.py` | Live webcam → ASCII renderer |
| `profile.json` | Auto-created on first run (your ID + username) |
| `chat_history.db` | Auto-created SQLite database |
