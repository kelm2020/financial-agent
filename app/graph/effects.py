"""Backend effects that leave an audit row (challenge point 10).

Every transfer to a person goes through ``transfer_to_human`` so the trace, the backend call and
the audit row cannot drift apart across the nodes that derive.
"""

from __future__ import annotations

import logging

from app.graph.context import GraphContext
from app.runtime.audit import AuditEventType
from app.tools.schemas import ToolResult, TransferResponse

_LOGGER = logging.getLogger("app.audit")


async def audit_effect(
    context: GraphContext,
    event_type: AuditEventType,
    *,
    conversation_id: str,
    **payload: str,
) -> bool:
    """Write one audit row and return whether it exists.

    Without a configured trail (offline unit tests) there is nothing to write. The caller decides
    what a missing row means: a write is not attempted; an effect that already happened is reported
    as ``audit_write_failed`` and logged by code.
    """
    if context.audit is None:
        return True
    try:
        await context.audit.record(
            event_type,
            conversation_id=conversation_id,
            customer_id=context.scope.customer_id,
            payload=payload,
        )
    except Exception:
        context.recorder.record_event("audit_write_failed", audit_event=event_type)
        _LOGGER.error("audit write failed event=%s", event_type)
        return False
    return True


async def transfer_to_human(
    context: GraphContext,
    *,
    conversation_id: str,
    motivo: str,
    resumen: str,
) -> ToolResult[TransferResponse]:
    """The single path to a person: trace, backend call and audit row."""
    context.recorder.record_tool("request_human", motivo=motivo)
    result = await context.gateway.transfer_to_human(
        context.scope, conversation_id=conversation_id, motivo=motivo, resumen=resumen
    )
    await audit_effect(
        context,
        "transfer_requested",
        conversation_id=conversation_id,
        motivo=motivo,
        status=result.status,
    )
    return result
