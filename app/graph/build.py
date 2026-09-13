from __future__ import annotations

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


def _after_hydrate(state: AgentState) -> str:
    if state.get("agreement_status") == "unknown":
        return "reconcile"
    if state.get("pending_draft") is not None:
        return "confirm"
    route = state.get("route_result")
    if (
        route is not None
        and route.intent == "aceptar_opcion"
        and state.get("guard_verdict") != "restrict"
    ):
        return "draft"
    if route is not None and route.intent == "pedido_humano":
        return "escalate"
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
    builder.add_node("guard_rules", guard_rules)
    builder.add_node("guard_classifier", guard_classifier)
    builder.add_node("route_or_confirm", route_or_confirm)
    builder.add_node("resolve_guard", resolve_guard_node)
    builder.add_node("deflect_plan", deflect_plan)
    builder.add_node("hydrate", hydrate)
    builder.add_node("build_draft", build_draft)
    builder.add_node("confirm_gate", confirm_gate)
    builder.add_node("execute_agreement", execute_agreement)
    builder.add_node("reconcile_agreement", reconcile_agreement)
    builder.add_node("plan_response", plan_from_route)
    builder.add_node("escalate", escalate)
    builder.add_node("render_and_validate", render_and_validate)
    builder.add_node("compact_context", compact_context)

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
