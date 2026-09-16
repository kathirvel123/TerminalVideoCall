# Terminal P2P Chat — Backend (Signaling Server)

A lightweight FastAPI WebSocket server that brokers WebRTC connections between clients.
**It never sees or stores any chat data, images, or video.**

## What it does
1. Accepts WebSocket connections from clients at `/ws/<user_id>`
2. Maintains an in-memory registry of online users
3. Forwards tiny WebRTC handshake messages (SDP offers/answers, ICE candidates) between peers
4. Once clients connect P2P, the server goes idle for that session

## Deploy on Render (free tier)

1. Push this `backend/` folder to a GitHub repo (or the root of your repo)
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your repo — Render auto-detects `render.yaml`
4. Click **Deploy** — your server will be live at `wss://your-app.onrender.com`

## Run locally (for development)

```bash
pip install -r requirements.txt
uvicorn signaling_server:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Variables
None required. The server is stateless.
