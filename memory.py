import os
from datetime import datetime
from sqlalchemy import create_engine, text
import config

DB_URL = os.environ.get("SQL_SERVER_CONNECTION_STRING")

if not DB_URL:
    raise ValueError(
        "SQL_SERVER_CONNECTION_STRING environment variable not set.\n"
        "Example: mssql+pyodbc://sa:password@localhost/AlzheimerRAG?driver=ODBC+Driver+17+for+SQL+Server"
    )
engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    future=True  # SQLAlchemy 2.0 style
)

def init_db() -> None:
    #Verify that SQL Server is reachable
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME IN ('users', 'conversations', 'messages')
        """))
        table_count = result.scalar()
        
    if table_count < 3:
        raise RuntimeError(
            f"SQL Server is connected, but only {table_count}/3 required tables were found. "
            "Please run the CREATE TABLE script in SSMS before starting the app."
        )


def get_or_create_user(user_id: str) -> None:
#Insert user only if they don't already exist.
    with engine.begin() as conn:
        conn.execute(
            text("""
                IF NOT EXISTS (SELECT 1 FROM users WHERE user_id = :user_id)
                    INSERT INTO users (user_id, name) VALUES (:user_id, :name)
            """),
            {"user_id": user_id, "name": user_id}
        )

def create_conversation(user_id: str) -> int:
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO conversations (user_id, started_at)
                OUTPUT INSERTED.conversation_id
                VALUES (:user_id, :started_at)
            """),
            {"user_id": user_id, "started_at": datetime.now()}
        )
        return result.scalar()


def get_latest_conversation_id(user_id: str):
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT TOP 1 conversation_id 
                FROM conversations
                WHERE user_id = :user_id
                ORDER BY started_at DESC
            """),
            {"user_id": user_id}
        )
        row = result.fetchone()
        return row[0] if row else None


def add_message(conversation_id: int, role: str, content: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO messages (conversation_id, role, content, timestamp)
                VALUES (:conversation_id, :role, :content, :timestamp)
            """),
            {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "timestamp": datetime.now()
            }
        )


def get_recent_messages(conversation_id: int, limit: int = None) -> list[dict]:
    """
    Fetch the N most recent messages and return them in chronological order
    (oldest first, newest last) so the LLM prompt reads naturally.
    """
    limit = limit or (config.MAX_HISTORY_TURNS * 2)
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT role, content FROM (
                    SELECT TOP (:limit) role, content, message_id
                    FROM messages
                    WHERE conversation_id = :conversation_id
                    ORDER BY message_id DESC
                ) AS recent
                ORDER BY message_id ASC
            """),
            {"conversation_id": conversation_id, "limit": limit}
        )
        rows = result.fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]


def list_conversations(user_id: str) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT conversation_id, started_at
                FROM conversations
                WHERE user_id = :user_id
                ORDER BY started_at DESC
            """),
            {"user_id": user_id}
        )
        rows = result.fetchall()
        return [{"conversation_id": r[0], "started_at": r[1]} for r in rows]


def get_conversation_preview(conversation_id: int) -> str:
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT TOP 1 content
                FROM messages
                WHERE conversation_id = :conversation_id AND role = 'user'
                ORDER BY message_id ASC
            """),
            {"conversation_id": conversation_id}
        )
        row = result.fetchone()
        return row[0] if row else "New conversation"


def get_all_messages(conversation_id: int) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT role, content
                FROM messages
                WHERE conversation_id = :conversation_id
                ORDER BY message_id ASC
            """),
            {"conversation_id": conversation_id}
        )
        rows = result.fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

class ConversationMemory:
    
    def __init__(self, user_id: str = "default_user", conversation_id: int = None, max_turns: int = None):
        init_db()                       # Verify DB is ready
        get_or_create_user(user_id)      # Ensure user row exists
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
        """Start a brand-new conversation. Old ones stay in the database."""
        self.conversation_id = create_conversation(self.user_id)