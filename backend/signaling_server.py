import json
import logging
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="P2P Terminal Chat Signaling Server")

# Store active connections: { "user_id": WebSocket }
active_connections: Dict[str, WebSocket] = {}


@app.get("/")
async def root():
    return {"message": "Signaling server is running"}


@app.get("/presence/{user_id}")
async def check_presence(user_id: str):
    """Check if a specific user is currently online."""
    return {"user_id": user_id, "online": user_id in active_connections}


@app.get("/online")
async def list_online():
    """Return a list of all currently connected user IDs."""
    return {"online_users": list(active_connections.keys())}


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    
    if user_id in active_connections:
        # If user is already connected, close the old connection
        old_ws = active_connections[user_id]
        try:
            await old_ws.close()
        except Exception:
            pass
            
    active_connections[user_id] = websocket
    logger.info(f"User {user_id} connected. Total users: {len(active_connections)}")

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON from {user_id}")
                continue

            # Expecting messages of format:
            # {
            #   "target_id": "other_user",
            #   "type": "offer" | "answer" | "ice" | "call_request",
            #   "payload": { ... SDP or ICE data ... }
            # }
            
            target_id = message.get("target_id")
            if not target_id:
                logger.warning(f"Message from {user_id} missing target_id")
                continue

            target_ws = active_connections.get(target_id)
            if target_ws:
                # Forward the message to the target, adding the sender's ID
                forward_msg = {
                    "sender_id": user_id,
                    "type": message.get("type"),
                    "payload": message.get("payload")
                }
                await target_ws.send_text(json.dumps(forward_msg))
                logger.info(f"Forwarded {message.get('type')} from {user_id} to {target_id}")
            else:
                # Inform sender that the target is offline
                error_msg = {
                    "sender_id": "server",
                    "type": "error",
                    "payload": {"message": f"User {target_id} is offline"}
                }
                await websocket.send_text(json.dumps(error_msg))
                logger.info(f"{user_id} tried to message offline user {target_id}")

    except WebSocketDisconnect:
        logger.info(f"User {user_id} disconnected")
        if user_id in active_connections:
            del active_connections[user_id]
    except Exception as e:
        logger.error(f"Error handling connection for {user_id}: {e}")
        if user_id in active_connections:
            del active_connections[user_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
