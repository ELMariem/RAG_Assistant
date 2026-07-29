# conversation buffer: Long-term conversation memory.

import sqlite3
import os
from datetime import datetime
import config

DB_PATH = os.path.join(config.BASE_DIR, "memory.db")


def init_db() -> None:
    """Create the tables if they don't exist yet. Safe to call every startup."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        )
    """)
    conn.commit()
    conn.close()


def get_or_create_user(user_id: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO users (user_id, name) VALUES (?, ?)", (user_id, user_id))
    conn.commit()
    conn.close()


def create_conversation(user_id: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (user_id, started_at) VALUES (?, ?)",
        (user_id, datetime.now().isoformat())
    )
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return conversation_id


def get_latest_conversation_id(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT conversation_id FROM conversations WHERE user_id = ? ORDER BY started_at DESC LIMIT 1",
        (user_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def add_message(conversation_id: int, role: str, content: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_recent_messages(conversation_id: int, limit: int = None) -> list[dict]:
    limit = limit or config.MAX_HISTORY_TURNS * 2  # *2: each turn = 1 user + 1 assistant message
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY message_id DESC LIMIT ?",
        (conversation_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    rows.reverse()  # back to chronological order (oldest first)
    return [{"role": r[0], "content": r[1]} for r in rows]


def list_conversations(user_id: str) -> list[dict]:
    #List past conversations for a user
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT conversation_id, started_at FROM conversations WHERE user_id = ? ORDER BY started_at DESC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"conversation_id": r[0], "started_at": r[1]} for r in rows]


class ConversationMemory:
    #public interface backed by SQLite history survives app restarts.

    def __init__(self, user_id: str = "default_user", conversation_id: int = None,  max_turns: int = None):
        init_db()
        get_or_create_user(user_id)
        self.user_id = user_id
        self.conversation_id = conversation_id or create_conversation(user_id)
        self.max_turns = max_turns or config.MAX_HISTORY_TURNS

    def add_turn(self, question: str, answer: str) -> None:
        add_message(self.conversation_id, "user", question)
        add_message(self.conversation_id, "assistant", answer)

    def get_history_text(self, max_turns: int = None) -> str:
        limit = (max_turns or self.max_turns) * 2
        messages = get_recent_messages(self.conversation_id, limit=limit)
        if not messages:
            return ""
        lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in messages]
        return "\n".join(lines)

    def clear(self) -> None:
        #Start a brand-new conversation. Old ones stay in the database, not deleted.
        self.conversation_id = create_conversation(self.user_id)