"""
core/conversation_context.py
=============================
ConversationContext — per-session state held in memory between HTTP requests.
Not persisted here; persistence lives in memory/sqlite_store.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationContext:
    """
    In-memory conversation state for one user session.
    Bounded to MAX_HISTORY turns to avoid bloating the LLM context window.
    """

    session_id: str
    user_id:    str
    department: str                              # "marketing" | "hr" | "finance"
    history:    list[dict] = field(default_factory=list)
    entity_memory: dict[str, Any] = field(default_factory=dict)

    MAX_HISTORY: int = field(default=20, init=False, repr=False)

    # ── Turn management ───────────────────────────────────────────────────────

    def add_turn(
        self,
        role:     str,
        content:  str,
        metadata: dict | None = None,
    ) -> None:
        """Append a role/content pair. Trims history to MAX_HISTORY on overflow."""
        turn: dict = {"role": role, "content": content}
        if metadata:
            turn["metadata"] = metadata
        self.history.append(turn)
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY :]

    def get_llm_messages(self, last_n: int = 10) -> list[dict]:
        """
        Returns the last n turns as {role, content} dicts suitable for passing
        to an Anthropic messages API call.
        Strips metadata — the API only accepts role/content.
        """
        msgs = [
            {"role": h["role"], "content": h["content"]}
            for h in self.history
            if h["role"] in ("user", "assistant")
        ]
        return msgs[-last_n:]

    def get_recent_context(self, n: int = 6) -> list[dict]:
        """Alias used by agents that want a shorter context slice."""
        return self.get_llm_messages(last_n=n)

    # ── Entity memory ─────────────────────────────────────────────────────────

    def set_entity(self, entity_type: str, value: Any) -> None:
        """
        Stores an extracted entity (e.g. {"region": "Pacific Northwest", "zip": "98101"}).
        Entity memory persists across turns within a session.
        """
        self.entity_memory[entity_type] = value

    def get_entity(self, entity_type: str, default: Any = None) -> Any:
        return self.entity_memory.get(entity_type, default)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self.history.clear()
        self.entity_memory.clear()

    @property
    def turn_count(self) -> int:
        return len([h for h in self.history if h["role"] == "user"])
