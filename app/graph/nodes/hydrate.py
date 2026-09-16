from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from langgraph.runtime import Runtime

from app.graph.context import GraphContext
from app.graph.state import AgentState, ResponsePlan
from app.guards.injection import mentions_foreign_customer
from app.obs.tracing import record_attribute, span
from app.policy.engine import opciones_permitidas
from app.tools.schemas import Customer, Debt, OptionsSnapshot, PaymentOption

# §8.1: customer/debt are cached per conversation for 15 minutes. Anything that leads to a
# write (build_draft, execute_agreement) ignores this cache and reads fresh (§8.3).
CACHE_TTL = timedelta(minutes=15)


def debt_fingerprint(debt: Debt) -> str:
    return hashlib.sha256(debt.model_dump_json().encode()).hexdigest()


@dataclass(slots=True)
class BusinessRead:
    customer: Customer | None = None
    debt: Debt | None = None
    debt_status: str = "unavailable"
    fingerprint: str = ""
    options: list[PaymentOption] | None = None
    updates: dict[str, object] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return (
            self.customer is not None
            and self.debt is not None
            and self.options is not None
            and bool(self.fingerprint)
        )


async def _not_requested() -> None:
    return None


async def read_business_data(
    context: GraphContext,
    *,
    customer: bool,
    debt: bool,
    options: bool,
    now: datetime,
    cached_customer: Customer | None = None,
    state: AgentState | None = None,
) -> BusinessRead:
    """Read the requested resources from the backend and return the state updates."""
    read = BusinessRead(customer=cached_customer)
    if customer:
        context.recorder.record_tool("get_customer", incluir_contacto=False)
    if debt or options:
        context.recorder.record_tool("get_debt", incluir_historial=False)
    if options:
        context.recorder.record_tool("get_payment_options", incluir_detalle=True)
    # The three reads are independent of each other: one backend round trip instead of three
    # (§12.4). The options snapshot only needs the debt fingerprint for INV-13, and that
    # fingerprint is computed locally from the debt result once it returns; the backend does
    # not need it.
    with span(
        "hydrate.business_reads",
        attributes={
            "app.customer_id": context.scope.customer_id,
            "tools.planned": sum([customer, debt, options]),
        },
    ) as hydrate_span:
        customer_result, debt_result, options_result = await asyncio.gather(
            context.gateway.get_customer(context.scope) if customer else _not_requested(),
            context.gateway.get_debt(context.scope) if debt or options else _not_requested(),
            context.gateway.get_payment_options(context.scope) if options else _not_requested(),
        )
        executed = sum(1 for r in (customer_result, debt_result, options_result) if r is not None)
        record_attribute(hydrate_span, "tools.executed", executed)
    if customer_result is not None and customer_result.status == "ok" and customer_result.data:
        read.customer = customer_result.data
        read.updates["customer"] = customer_result.data
    if debt_result is not None:
        if debt_result.status == "ok" and debt_result.data is not None:
            read.debt = debt_result.data
            read.debt_status = "ok"
            read.fingerprint = debt_fingerprint(debt_result.data)
            read.updates.update(
                {
                    "debt": debt_result.data,
                    "debt_status": "ok",
                    "debt_fingerprint": read.fingerprint,
                    "debt_fetched_at": now,
                }
            )
        else:
            read.debt_status = "not_found" if debt_result.status == "not_found" else "unavailable"
            read.updates["debt_status"] = read.debt_status
            if read.debt_status == "unavailable" and (
                state is None or state.get("debt") is not None
            ):
                # A record whose refresh just failed is stale, not current: it must not come
                # back as "al día de hoy" nor feed decisions that need fresh data (§5.1
                # as_of). The dependent options are invalidated with it, the same way a
                # fingerprint change does (INV-13).
                read.updates.update(
                    {
                        "debt": None,
                        "debt_fingerprint": "",
                        "debt_fetched_at": None,
                        "options_snapshot": None,
                        "offered_options": [],
                    }
                )
    if options and options_result is not None and read.fingerprint:
        if options_result.status == "ok" and options_result.data is not None:
            read.options = list(options_result.data.opciones)
            read.updates["options_snapshot"] = OptionsSnapshot(
                options=read.options, fetched_at=now, debt_fingerprint=read.fingerprint
            )
    return read


async def hydrate(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, object]:
    context = runtime.context
    if state.get("guard_verdict") == "restrict" and mentions_foreign_customer(
        state.get("detection_text", ""), context.scope.customer_id
    ):
        # A foreign identifier is answered without touching even the authenticated account.
        # This makes the IDOR defense visible in the tool trajectory, not only in URL scoping.
        return {}
    now = context.clock.now()
    route = state.get("route_result")
    intent = route.intent if route is not None else None
    needs_debt = intent in {
        "consulta_deuda",
        "negociacion",
    }
    needs_options = intent == "negociacion"
    # The balance reply offers alternatives only when the policy would allow them (identity,
    # segment, broken plans), which needs the customer as well.
    needs_customer = needs_options or intent == "consulta_deuda"

    fetched_at = state.get("debt_fetched_at")
    debt_is_fresh = (
        state.get("debt") is not None and fetched_at is not None and now - fetched_at < CACHE_TTL
    )
    snapshot = state.get("options_snapshot")
    snapshot_is_fresh = (
        isinstance(snapshot, OptionsSnapshot)
        and debt_is_fresh
        and snapshot.debt_fingerprint == state.get("debt_fingerprint")
        and now - snapshot.fetched_at < CACHE_TTL
    )
    read = await read_business_data(
        context,
        customer=needs_customer and state.get("customer") is None,
        debt=needs_debt and not debt_is_fresh,
        options=needs_options and not snapshot_is_fresh,
        now=now,
        state=state,
    )
    updates = dict(read.updates)
    if not needs_options:
        return updates
    customer = read.customer or state.get("customer")
    debt = read.debt or state.get("debt")
    backend = read.options
    if backend is None and snapshot_is_fresh:
        assert isinstance(snapshot, OptionsSnapshot)
        backend = list(snapshot.options)
    if backend is None or customer is None or debt is None:
        updates["response_plan"] = ResponsePlan(kind="error", template_id="data_unavailable")
        return updates
    # Only what the policy engine allows at this instant is ever presented (§8.1 negotiate).
    updates["offered_options"] = opciones_permitidas(customer, debt, backend, as_of=now)
    return updates
