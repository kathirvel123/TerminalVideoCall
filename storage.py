import json
import os
import sqlite3
import uuid
from typing import List, Dict, Optional
from datetime import datetime

PROFILE_FILE = "profile.json"
DB_FILE = "chat_history.db"

class StorageManager:
    def __init__(self):
        self.profile = self._load_or_create_profile()
        self.conn = sqlite3.connect(DB_FILE)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _load_or_create_profile(self) -> dict:
        if os.path.exists(PROFILE_FILE):
            try:
                with open(PROFILE_FILE, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        
        # Create a new profile if none exists or it's invalid
        new_profile = {
            "user_id": str(uuid.uuid4())[:8],  # short unique ID
            "username": f"User_{str(uuid.uuid4())[:4]}"
        }
        self.save_profile(new_profile)
        return new_profile

    def save_profile(self, profile: dict):
        self.profile = profile
        with open(PROFILE_FILE, "w") as f:
            json.dump(self.profile, f, indent=4)

    def get_user_id(self) -> str:
        return self.profile.get("user_id", "")

    def get_username(self) -> str:
        return self.profile.get("username", "")

    def update_username(self, new_username: str):
        self.profile["username"] = new_username
        self.save_profile(self.profile)

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                peer_id TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                message_type TEXT NOT NULL, -- 'text', 'image_ascii'
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def save_message(self, peer_id: str, sender_id: str, message_type: str, content: str):
        """
        Save a message to the local SQLite database.
        `peer_id` is the ID of the person you are chatting with (used for grouping).
        `sender_id` is who actually sent the message (could be you or the peer).
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO messages (peer_id, sender_id, message_type, content)
            VALUES (?, ?, ?, ?)
        ''', (peer_id, sender_id, message_type, content))
        self.conn.commit()

    def get_chat_history(self, peer_id: str, limit: int = 50) -> List[Dict]:
        """
        Retrieve chat history with a specific peer.
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT sender_id, message_type, content, timestamp
            FROM messages
            WHERE peer_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (peer_id, limit))
        
        rows = cursor.fetchall()
        # Return in chronological order
        return [dict(row) for row in reversed(rows)]

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    # Test the storage manager
    store = StorageManager()
    print(f"Loaded Profile: {store.profile}")
    store.save_message("peer_123", store.get_user_id(), "text", "Hello, World!")
    store.save_message("peer_123", "peer_123", "text", "Hi there!")
    history = store.get_chat_history("peer_123")
    print("\nChat History:")
    for msg in history:
        print(f"[{msg['timestamp']}] {msg['sender_id']}: {msg['content']}")
    store.close()
