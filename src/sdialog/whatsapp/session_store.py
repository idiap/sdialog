# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-FileContributor: Amit Bhakt <amitbhakt@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""
SQLite-backed multi-user session and idempotency store for WhatsApp.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional, Any, Dict, List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

logger = logging.getLogger(__name__)

_TYPE_TO_MSG_CLASS = {
    "HumanMessage": HumanMessage,
    "AIMessage": AIMessage,
    "SystemMessage": SystemMessage,
}


class SQLiteSessionStore:
    """
    Thread-safe SQLite store for WhatsApp user sessions and webhook idempotency.

    Features:
    - WAL mode for concurrent non-blocking reads/writes.
    - Idempotency tracking to reject duplicate Meta webhook deliveries.
    - Automatic session expiration after inactivity timeout.
    """

    def __init__(self, db_path: str = "whatsapp_sessions.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with WAL mode and busy timeout configured."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Initialize database tables and indexes."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS whatsapp_sessions (
                    wa_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS processed_messages (
                    wamid TEXT PRIMARY KEY,
                    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_processed_received
                ON processed_messages(received_at);
            """)

    def load_session(self, wa_id: str, timeout_seconds: int = 1800) -> Optional[Dict[str, Any]]:
        """
        Load session state for a user.

        Returns None if session does not exist or has exceeded timeout_seconds of inactivity.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT state, last_active FROM whatsapp_sessions WHERE wa_id = ?",
                (wa_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            state_json, last_active_str = row["state"], row["last_active"]

            # Parse last_active timestamp
            try:
                # Handle SQLite timestamp formats
                if "T" in last_active_str:
                    last_active = datetime.fromisoformat(last_active_str)
                else:
                    last_active = datetime.strptime(last_active_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                last_active = datetime.now()

            # Check timeout
            elapsed = (datetime.now() - last_active).total_seconds()
            if elapsed > timeout_seconds:
                logger.info(f"[WhatsApp] Session for user {wa_id} expired ({elapsed:.0f}s > {timeout_seconds}s)")
                self.reset_session(wa_id)
                return None

            try:
                return json.loads(state_json)
            except json.JSONDecodeError:
                logger.warning(f"[WhatsApp] Corrupted session JSON for {wa_id}, resetting")
                self.reset_session(wa_id)
                return None

    def save_session(self, wa_id: str, state: Dict[str, Any]) -> None:
        """Save or update the session state for a user with the current timestamp."""
        state_json = json.dumps(state)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO whatsapp_sessions (wa_id, state, last_active)
                VALUES (?, ?, ?)
                ON CONFLICT(wa_id) DO UPDATE SET
                    state = excluded.state,
                    last_active = excluded.last_active;
                """,
                (wa_id, state_json, now_str)
            )

    def reset_session(self, wa_id: str) -> None:
        """Reset and remove session state for a user."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM whatsapp_sessions WHERE wa_id = ?", (wa_id,))
            logger.info(f"[WhatsApp] Session reset for {wa_id}")

    def is_message_processed(self, wamid: str) -> bool:
        """Check whether a message ID has already been received and processed."""
        if not wamid:
            return False
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM processed_messages WHERE wamid = ?",
                (wamid,)
            )
            return cursor.fetchone() is not None

    def mark_message_processed(self, wamid: str) -> None:
        """Record a message ID in the idempotency table."""
        if not wamid:
            return
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO processed_messages (wamid, received_at) VALUES (?, ?)",
                (wamid, now_str)
            )

    def cleanup_old_records(self, retention_days: int = 7) -> None:
        """Purge idempotency records older than retention_days."""
        with self._get_connection() as conn:
            conn.execute(
                f"DELETE FROM processed_messages WHERE received_at < datetime('now', '-{retention_days} days')"
            )


# ---------------------------------------------------------------------------
# State Bridge: Supports both future Agent.get_state() and interim fallback
# ---------------------------------------------------------------------------

def restore_agent_state(agent: Any, state: Optional[Dict[str, Any]]) -> None:
    """Restore agent state. Prefers native agent.set_state() if present."""
    if not state:
        return

    if hasattr(agent, "set_state") and callable(agent.set_state):
        agent.set_state(state)
        return

    # Fallback bridge before core SDialog implements Agent.set_state():
    memory_data = state.get("memory", [])
    restored_msgs: List[BaseMessage] = []
    for item in memory_data:
        msg_cls = _TYPE_TO_MSG_CLASS.get(item.get("type"), HumanMessage)
        restored_msgs.append(msg_cls(content=item.get("content", "")))

    if restored_msgs:
        # Keep initial system prompt / persona prompt if not present in saved turns
        if agent.memory and isinstance(agent.memory[0], SystemMessage):
            agent.memory[:] = [agent.memory[0]] + restored_msgs
        else:
            agent.memory[:] = restored_msgs


def dump_agent_state(agent: Any, max_history_turns: int = 40) -> Dict[str, Any]:
    """Serialize agent state. Prefers native agent.get_state() if present."""
    if hasattr(agent, "get_state") and callable(agent.get_state):
        return agent.get_state()

    # Fallback bridge before core SDialog implements Agent.get_state():
    memory_items = []
    # Skip the primary system message (turn 0) in the dynamic history
    history = agent.memory[1:] if (agent.memory and isinstance(agent.memory[0], SystemMessage)) else agent.memory

    # Apply sliding window budget
    if len(history) > max_history_turns:
        history = history[-max_history_turns:]

    for msg in history:
        memory_items.append({
            "type": msg.__class__.__name__,
            "content": msg.content
        })

    return {
        "version": 1,
        "memory": memory_items,
        "finished": getattr(agent, "_finished", False)
    }
