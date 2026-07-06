"""
memory/sqlite_store.py
======================
SQLiteContextStore — async SQLite persistence layer for:
  • Conversation turns  (per user × session)
  • Entity memory       (persistent facts extracted from conversations)
  • Tool result cache   (TTL-based cache to avoid redundant API calls)

Uses aiosqlite so it plays nicely with the async FastAPI + LangGraph stack.
"""

from __future__ import annotations
import json
from datetime import datetime
from typing import Any

import aiosqlite


class SQLiteContextStore:
    """
    Thread-safe async SQLite store.
    Call `await store.initialize()` once at startup before any reads/writes.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Creates tables and indexes on first run; idempotent on subsequent runs."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT    NOT NULL,
                    session_id  TEXT    NOT NULL,
                    query       TEXT    NOT NULL,
                    response    TEXT    NOT NULL,
                    agent       TEXT    NOT NULL,
                    metadata    TEXT    NOT NULL DEFAULT '{}',
                    created_at  TEXT    NOT NULL
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_user_session
                ON conversation_turns (user_id, session_id, id DESC)
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS entity_memory (
                    user_id     TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    value       TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    PRIMARY KEY (user_id, entity_type)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS tool_cache (
                    cache_key   TEXT PRIMARY KEY,
                    result      TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    ttl_seconds INTEGER NOT NULL DEFAULT 3600
                )
            """)

            await db.commit()

    # ── Conversation turns ────────────────────────────────────────────────────

    async def save_turn(
        self,
        user_id:    str,
        session_id: str,
        query:      str,
        response:   str,
        agent:      str,
        metadata:   dict | None = None,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO conversation_turns
                   (user_id, session_id, query, response, agent, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id, session_id, query, response, agent,
                    json.dumps(metadata or {}),
                    datetime.utcnow().isoformat(),
                ),
            )
            await db.commit()

    async def get_context(
        self,
        user_id:    str,
        session_id: str,
        limit:      int = 10,
    ) -> list[dict]:
        """Returns the last `limit` turns in chronological order."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT query, response, agent, metadata, created_at
                   FROM conversation_turns
                   WHERE user_id = ? AND session_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (user_id, session_id, limit),
            )
            rows = await cursor.fetchall()

        # Reverse so oldest is first (chronological for LLM context)
        return [
            {
                "query":      r["query"],
                "response":   r["response"],
                "agent":      r["agent"],
                "metadata":   json.loads(r["metadata"]),
                "created_at": r["created_at"],
            }
            for r in reversed(rows)
        ]

    async def get_all_sessions(self, user_id: str) -> list[str]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT DISTINCT session_id FROM conversation_turns WHERE user_id = ? ORDER BY id DESC",
                (user_id,),
            )
            rows = await cursor.fetchall()
        return [r["session_id"] for r in rows]

    # ── Entity memory ─────────────────────────────────────────────────────────

    async def get_entities(self, user_id: str) -> dict[str, Any]:
        """Returns all stored entities for a user as {entity_type: value}."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT entity_type, value FROM entity_memory WHERE user_id = ?",
                (user_id,),
            )
            rows = await cursor.fetchall()
        return {r["entity_type"]: json.loads(r["value"]) for r in rows}

    async def set_entity(
        self,
        user_id:     str,
        entity_type: str,
        value:       Any,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO entity_memory
                   (user_id, entity_type, value, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (user_id, entity_type, json.dumps(value, default=str), datetime.utcnow().isoformat()),
            )
            await db.commit()

    # ── Tool result cache ─────────────────────────────────────────────────────

    async def get_cached(self, cache_key: str) -> Any | None:
        """Returns cached tool result if not expired, else None."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT result, created_at, ttl_seconds FROM tool_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = await cursor.fetchone()

        if not row:
            return None

        age = (datetime.utcnow() - datetime.fromisoformat(row["created_at"])).total_seconds()
        if age > row["ttl_seconds"]:
            return None
        return json.loads(row["result"])

    async def set_cached(
        self,
        cache_key:   str,
        result:      Any,
        ttl_seconds: int = 3600,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO tool_cache
                   (cache_key, result, created_at, ttl_seconds)
                   VALUES (?, ?, ?, ?)""",
                (cache_key, json.dumps(result, default=str), datetime.utcnow().isoformat(), ttl_seconds),
            )
            await db.commit()

    async def invalidate_cache(self, cache_key: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM tool_cache WHERE cache_key = ?", (cache_key,))
            await db.commit()
