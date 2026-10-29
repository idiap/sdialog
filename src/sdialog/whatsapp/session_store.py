# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Amit Bhakt <amitbhakt@users.noreply.github.com>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
"""
SQLite-backed multi-user session and idempotency store for WhatsApp.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, Iterator

logger = logging.getLogger(__name__)


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

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Open a SQLite connection (WAL mode, busy timeout), commit on success and always close it."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            with conn:
                yield conn
        finally:
            conn.close()

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

    def load_session(self, wa_id: str, timeout_seconds: Optional[int] = 1800) -> Optional[Dict[str, Any]]:
        """
        Load session state for a user.

        Returns None if session does not exist or has exceeded timeout_seconds of inactivity
        (sessions never expire if timeout_seconds is None).
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
            if timeout_seconds is not None and elapsed > timeout_seconds:
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

    def cleanup_old_records(self, retention_days: int = 7, session_retention_days: Optional[int] = None) -> None:
        """
        Purge idempotency records older than retention_days and, if session_retention_days is set,
        the sessions of users inactive for longer than that.
        """
        # Timestamps are stored in local time, so compute the cutoffs the same way (SQLite's 'now' is UTC)
        def cutoff(days: int) -> str:
            return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            conn.execute("DELETE FROM processed_messages WHERE received_at < ?", (cutoff(retention_days),))
            if session_retention_days is not None:
                deleted = conn.execute("DELETE FROM whatsapp_sessions WHERE last_active < ?",
                                       (cutoff(session_retention_days),)).rowcount
                if deleted:
                    logger.info(f"[WhatsApp] Deleted {deleted} session(s) inactive for over "
                                f"{session_retention_days} days")
