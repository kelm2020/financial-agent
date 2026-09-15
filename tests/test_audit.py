"""Audit trail of confirmed writes, their outcomes and transfers (challenge point 10)."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
from pydantic import SecretStr

from app.graph.nodes.respond import _STATIC_TEMPLATES
from app.main import _audit_key
from app.runtime.audit import AuditRecord, AuditTrail, InMemoryAuditLog
from tests.agent_support import BackendFault, agent_runtime, fixture_draft, offline_settings


class BrokenAuditLog:
    async def append(self, record: AuditRecord) -> None:
        raise RuntimeError("audit store down")


def _broken(runtime: Any) -> Any:
    return replace(runtime.context, audit=AuditTrail(BrokenAuditLog(), key=b"key"))


async def _conversation_with_draft(runtime: Any) -> Any:
    conversation = await runtime.service.create_conversation("CUST-00125")
    await runtime.seed(conversation, {"pending_draft": await fixture_draft(runtime)})
    return conversation


async def _say(runtime: Any, conversation: Any, text: str, context: Any = None) -> Any:
    return await runtime.service.send_message(
        conversation.conversation_id,
        conversation.customer_id,
        text,
        context=context or runtime.context,
    )


def _agreement_posts(runtime: Any) -> int:
    return sum(
        1
        for method, path in runtime.transport.requests
        if (method, path) == ("POST", "/payment-agreement")
    )


def test_records_are_sealed_and_the_customer_is_pseudonymized() -> None:
    trail = AuditTrail(InMemoryAuditLog(), key=b"key")
    reference = trail.customer_ref("CUST-00125")
    assert reference != trail.customer_ref("CUST-00126")
    assert "CUST" not in reference and len(reference) == 64
    payload = {"motivo": "reclamo"}
    record = AuditRecord(
        "conversation",
        reference,
        "transfer_requested",
        payload,
        trail.seal("conversation", reference, "transfer_requested", payload),
    )
    assert trail.verifies(record)
    assert not trail.verifies(replace(record, payload={"motivo": "vulnerabilidad"}))
    assert not AuditTrail(InMemoryAuditLog(), key=b"other").verifies(record)
    with pytest.raises(ValueError, match="audit key"):
        AuditTrail(InMemoryAuditLog(), key=b"")


async def test_a_confirmed_agreement_is_audited_before_and_after_the_write() -> None:
    async with agent_runtime() as runtime:
        conversation = await _conversation_with_draft(runtime)
        await _say(runtime, conversation, "sí")
        records = list(runtime.audit_log.records)
        trail = runtime.context.audit
    assert [record.event_type for record in records] == [
        "agreement_write_requested",
        "agreement_created",
    ]
    assert trail is not None and all(trail.verifies(record) for record in records)
    assert records[0].payload["idempotency_key"] and records[0].payload["confirmation_event_id"]
    assert records[1].payload["agreement_id"].startswith("AGR-")
    assert {record.conversation_id for record in records} == {conversation.conversation_id}
    assert "CUST-00125" not in json.dumps([record.payload for record in records])


async def test_a_write_that_cannot_be_audited_is_not_attempted() -> None:
    async with agent_runtime() as runtime:
        conversation = await _conversation_with_draft(runtime)
        result = await _say(runtime, conversation, "sí", _broken(runtime))
        assert _agreement_posts(runtime) == 0 and not runtime.recorder.agreement_writes
        assert {
            "type": "audit_write_failed",
            "audit_event": "agreement_write_requested",
        } in runtime.recorder.events
    assert result.text == _STATIC_TEMPLATES["data_unavailable"]
    # The draft stays pending: a later "sí" retries under the same idempotency key.
    assert result.state["pending_draft"] is not None


async def test_an_unknown_outcome_is_audited_and_its_replay_needs_an_audit_row() -> None:
    fault = BackendFault("POST", "/payment-agreement", "timeout_after_commit")
    async with agent_runtime(faults=[fault]) as runtime:
        conversation = await _conversation_with_draft(runtime)
        await _say(runtime, conversation, "sí")
        audited = [record.event_type for record in runtime.audit_log.records]
        posts = _agreement_posts(runtime)
        retried = await _say(runtime, conversation, "¿quedó registrado?", _broken(runtime))
        assert _agreement_posts(runtime) == posts
    assert audited == [
        "agreement_write_requested",
        "agreement_outcome_unknown",
        "transfer_requested",
    ]
    assert retried.state["agreement_status"] == "unknown"


async def test_every_transfer_to_a_person_is_audited() -> None:
    async with agent_runtime() as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        await _say(runtime, conversation, "Quiero hablar con una persona")
        records = [(record.event_type, record.payload) for record in runtime.audit_log.records]
    assert records == [("transfer_requested", {"motivo": "pedido_explicito", "status": "ok"})]


def test_the_audit_key_is_explicit_or_stable_across_production_replicas() -> None:
    production = offline_settings(app_env="production")
    rotated = offline_settings(app_env="production", mock_token_secret=SecretStr("other"))
    local = offline_settings(app_env="local")
    assert _audit_key(offline_settings(audit_hmac_key=SecretStr("configured"))) == b"configured"
    assert _audit_key(production) == _audit_key(production) != _audit_key(rotated)
    assert _audit_key(local) != _audit_key(local)
