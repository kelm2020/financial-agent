from __future__ import annotations

import re

from app.guards.injection import INJECTION_PATTERNS
from app.guards.normalize import detection_skeleton, normalize_visible

# Imperatives addressed to the agent. A rolling summary (§8.4) describes the conversation; it
# never instructs the assistant, so any of these makes the summary untrusted for reuse.
_AGENT_DIRECTIVES = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:el|al) (?:asistente|agente|modelo|sistema) (?:debe|tiene que|deberia|puede)\b",
        r"\b(?:instruccion|instrucciones) (?:para|al) (?:el )?(?:asistente|agente|modelo)\b",
        r"\b(?:no )?(?:respetes|obedezcas|sigas) (?:las|tus) (?:reglas|politicas|instrucciones)\b",
        r"\b(?:registra|crea|aproba|confirma) (?:el|un) acuerdo sin (?:confirmacion|preguntar)\b",
    )
)


def spotlight(label: str, source_id: str, text: str, *, max_characters: int = 4_000) -> str:
    """Wrap untrusted data without allowing its contents to close the delimiter."""
    normalized = normalize_visible(text)[:max_characters]
    safe_label = "".join(
        character for character in label if character.isalnum() or character == "_"
    )
    safe_id = "".join(
        character for character in source_id if character.isalnum() or character in "-_"
    )
    # Neutralize every delimiter-like sequence, not only this label's closing tag: a chunk must
    # not be able to open or close any data block.
    normalized = re.sub(r"<<\s*/?\s*[A-Za-z_]+[^>]*>>", "[DELIMITADOR_NEUTRALIZADO]", normalized)
    closing = f"<</{safe_label}>>"
    return f"<<{safe_label} id={safe_id}>>\n{normalized}\n{closing}"


def summary_is_safe(summary: str) -> bool:
    """A model-written summary is untrusted text: it may describe, never instruct (§10.1.2)."""
    skeleton = detection_skeleton(summary)
    return not any(
        pattern.search(skeleton) for pattern in (*INJECTION_PATTERNS, *_AGENT_DIRECTIVES)
    )
