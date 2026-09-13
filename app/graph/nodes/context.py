from __future__ import annotations

from langchain_core.messages import AnyMessage, RemoveMessage

from app.graph.state import AgentState
from app.guards.normalize import normalize_visible
from app.guards.untrusted import summary_is_safe

_RETAINED_MESSAGES = 16  # eight user/assistant turns
_SUMMARY_INTERVAL = 6
_SUMMARY_LIMIT = 2_000


def _summary_line(message: AnyMessage) -> str:
    role = "cliente" if message.type == "human" else "asistente"
    content = message.content if isinstance(message.content, str) else ""
    return f"{role}: {normalize_visible(content)}"


async def compact_context(state: AgentState) -> dict[str, object]:
    """Keep eight turns and periodically fold evicted, validated messages into a safe summary."""
    messages = list(state.get("messages", []))
    evicted = messages[:-_RETAINED_MESSAGES]
    pending = [*state.get("summary_pending", []), *map(_summary_line, evicted)]
    since = state.get("turns_since_summary", 0) + 1
    updates: dict[str, object] = {
        "turns_since_summary": since,
        "summary_pending": pending,
    }
    removable = [RemoveMessage(id=message.id) for message in evicted if message.id]
    if removable:
        updates["messages"] = removable
    if since >= _SUMMARY_INTERVAL:
        previous = state.get("conversation_summary", "")
        candidate = "\n".join(part for part in (previous, *pending) if part)[-_SUMMARY_LIMIT:]
        if candidate and summary_is_safe(candidate):
            updates["conversation_summary"] = candidate
        elif previous and summary_is_safe(previous):
            updates["conversation_summary"] = previous
        elif pending:
            updates["conversation_summary"] = (
                "Hubo intercambios previos sin instrucciones reutilizables."
            )
        updates["summary_pending"] = []
        updates["turns_since_summary"] = 0
    return updates
