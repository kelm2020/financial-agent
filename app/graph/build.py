from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph.context import GraphContext
from app.graph.nodes.agreement import (
    build_draft,
    confirm_gate,
    execute_agreement,
    reconcile_agreement,
)
from app.graph.nodes.context import compact_context
from app.graph.nodes.guards import (
    guard_classifier,
    guard_path,
    guard_rules,
    resolve_guard_node,
    route_or_confirm,
)
from app.graph.nodes.hydrate import hydrate
from app.graph.nodes.respond import (
    deflect_plan,
    escalate,
    plan_from_route,
    render_and_validate,
)
from app.graph.state import AgentState


def _timed(
    name: str,
    fn: Callable[..., Awaitable[dict[str, object]]],
) -> Callable[..., Awaitable[dict[str, object]]]:
    """Wrap a graph node so its wall-clock time reaches the per-node latency breakdown.

    Accepts both two-arg ``fn(state, runtime)`` nodes (the common shape) and one-arg
    ``fn(state)`` nodes (like ``compact_context``).
    """
    import inspect

    takes_runtime = "runtime" in inspect.signature(fn).parameters

    if takes_runtime:

        async def with_runtime(state: AgentState, runtime: Any) -> dict[str, object]:
            recorder = getattr(runtime.context, "recorder", None)
            started = time.perf_counter()
            try:
                return await fn(state, runtime)
            finally:
                if recorder is not None:
                    recorder.record_node_latency(name, (time.perf_counter() - started) * 1000)

        return with_runtime

    async def without_runtime(state: AgentState, runtime: Any) -> dict[str, object]:
        recorder = getattr(runtime.context, "recorder", None)
        started = time.perf_counter()
        try:
            return await fn(state)
        finally:
            if recorder is not None:
                recorder.record_node_latency(name, (time.perf_counter() - started) * 1000)

    return without_runtime


def _after_hydrate(state: AgentState) -> str:
    if state.get("agreement_status") == "unknown":
        return "reconcile"
    route = state.get("route_result")
    # ESC-001/002 and an explicit request for a person always outrank a pending draft. A customer
    # can disclose vulnerability, raise a dispute or ask for help while reading the confirmation.
    if route is not None and route.intent == "pedido_humano":
        return "escalate"
    if state.get("pending_draft") is not None:
        return "confirm"
    if (
        route is not None
        and route.intent == "aceptar_opcion"
        and state.get("guard_verdict") != "restrict"
    ):
        return "draft"
    return "plan"


def _after_confirm(state: AgentState) -> str:
    # confirm_gate either wrote the final plan, or kept a valid draft for "yes" or "other".
    if state.get("response_plan") is not None:
        return "render"
    candidate = state.get("confirmation_candidate")
    if (
        candidate is not None
        and candidate.verdict == "yes"
        and state.get("guard_verdict") != "restrict"
    ):
        return "execute"
    return "answer"


def build_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Any:
    builder = StateGraph(AgentState, context_schema=GraphContext)
    builder.add_node("guard_rules", _timed("guard_rules", guard_rules))
    builder.add_node("guard_classifier", _timed("guard_classifier", guard_classifier))
    builder.add_node("route_or_confirm", _timed("route_or_confirm", route_or_confirm))
    builder.add_node("resolve_guard", _timed("resolve_guard", resolve_guard_node))
    builder.add_node("deflect_plan", _timed("deflect_plan", deflect_plan))
    builder.add_node("hydrate", _timed("hydrate", hydrate))
    builder.add_node("build_draft", _timed("build_draft", build_draft))
    builder.add_node("confirm_gate", _timed("confirm_gate", confirm_gate))
    builder.add_node("execute_agreement", _timed("execute_agreement", execute_agreement))
    builder.add_node("reconcile_agreement", _timed("reconcile_agreement", reconcile_agreement))
    builder.add_node("plan_response", _timed("plan_response", plan_from_route))
    builder.add_node("escalate", _timed("escalate", escalate))
    builder.add_node("render_and_validate", _timed("render_and_validate", render_and_validate))
    builder.add_node("compact_context", _timed("compact_context", compact_context))

    builder.add_edge(START, "guard_rules")
    builder.add_edge(START, "guard_classifier")
    builder.add_edge(START, "route_or_confirm")
    builder.add_edge(["guard_rules", "guard_classifier", "route_or_confirm"], "resolve_guard")
    builder.add_conditional_edges(
        "resolve_guard",
        guard_path,
        {"deflect": "deflect_plan", "continue": "hydrate"},
    )
    builder.add_edge("deflect_plan", "render_and_validate")
    builder.add_conditional_edges(
        "hydrate",
        _after_hydrate,
        {
            "confirm": "confirm_gate",
            "draft": "build_draft",
            "escalate": "escalate",
            "reconcile": "reconcile_agreement",
            "plan": "plan_response",
        },
    )
    builder.add_conditional_edges(
        "confirm_gate",
        _after_confirm,
        {
            "execute": "execute_agreement",
            "answer": "plan_response",
            "render": "render_and_validate",
        },
    )
    builder.add_edge("build_draft", "render_and_validate")
    builder.add_edge("execute_agreement", "render_and_validate")
    builder.add_edge("reconcile_agreement", "render_and_validate")
    builder.add_edge("plan_response", "render_and_validate")
    builder.add_edge("escalate", "render_and_validate")
    builder.add_edge("render_and_validate", "compact_context")
    builder.add_edge("compact_context", END)
    return builder.compile(checkpointer=checkpointer)
