from __future__ import annotations

from pathlib import Path

from evals.models import EvalReport, Rate


def _ratio(rate: Rate) -> str:
    value = rate.value if rate.denominator else 0.0
    return f"{rate.numerator}/{rate.denominator} ({value:.3f})"


def render_report(report: EvalReport) -> str:
    metrics = report.metrics
    judge = (
        f"{_ratio(metrics.quality_judged)} · {report.judge_model}"
        + (f" · {metrics.quality_unjudged} sin juzgar" if metrics.quality_unjudged else "")
        if metrics.quality_judged.denominator
        else "no ejecutado (usar --judge-model)"
    )
    lines = [
        f"F4 evaluation · suite={report.suite} · dataset={report.dataset} · k={report.k} · "
        f"base={report.base_cases} · expanded={report.expanded_cases}",
        f"Agente={report.agent_model or 'offline'} · prompt={report.prompt_fingerprint or 'n/d'}",
        "",
        "| Eje | Métrica | Resultado |",
        "|---|---|---:|",
        f"| Tools y trayectoria | tool_selection_f1 | {metrics.tool_selection_f1:.3f} |",
        f"| Tools y trayectoria | valid_tool_args | {_ratio(metrics.valid_tool_args)} |",
        f"| Tools y trayectoria | trajectory_match | {_ratio(metrics.trajectory_match)} |",
        f"| Grounding y retrieval | grounded_answers | {_ratio(metrics.grounded_answers)} |",
        "| Grounding y retrieval | model_answers_accepted | "
        f"{_ratio(metrics.model_answers_accepted)} |",
        f"| Grounding y retrieval | hallucinated_numbers | "
        f"{_ratio(metrics.hallucinated_numbers)} |",
        f"| Cumplimiento | policy_compliance | {_ratio(metrics.policy_compliance)} |",
        f"| Cumplimiento | unsafe_auto_action | {_ratio(metrics.unsafe_auto_action)} |",
        f"| Cumplimiento | confirmation_bypass | {_ratio(metrics.confirmation_bypass)} |",
        f"| Escalamiento | recall | {_ratio(metrics.escalation_recall)} |",
        f"| Escalamiento | precision | {_ratio(metrics.escalation_precision)} |",
        f"| Calidad conversacional | judge (todos los criterios pass) | {judge} |",
    ]
    lines.extend(
        f"| Calidad conversacional | {criterion} | {_ratio(rate)} |"
        for criterion, rate in metrics.quality_by_criterion.items()
    )
    latency = f"p95 turno: {metrics.p95_turn_latency_ms:.1f} ms"
    if report.concurrency > 1:
        # Cases waited on each other, so this number is contention, not the agent's latency.
        latency += f" (NO COMPARABLE: {report.concurrency} casos en paralelo)"
    lines.extend(
        (
            "",
            f"Casos: {_ratio(metrics.cases_passed)} · pass^k: {_ratio(report.pass_to_k)} · "
            f"{latency}",
        )
    )
    if metrics.total_cost_usd is not None:
        lines.append(f"Costo total: USD {metrics.total_cost_usd:.6f}")
    if metrics.input_tokens or metrics.output_tokens:
        lines.append(
            f"Tokens: entrada={metrics.input_tokens} · cacheados={metrics.cached_tokens} · "
            f"salida={metrics.output_tokens}"
        )
    failed = [result for result in metrics.results if not result.passed]
    if failed:
        lines.extend(("", "Fallos:"))
        for result in failed:
            lines.append(f"- {result.case_id}: {'; '.join(result.failures)}")
            lines.extend(f"    {line}" for line in result.diagnostics)
    lines.append("")
    lines.append(
        (
            "Gates (sólo seguridad: nivel A sin modelo): "
            if report.gate_profile == "safety"
            else "Gates: "
        )
        + (", ".join(metrics.gate_failures) if metrics.gate_failures else "PASS")
    )
    return "\n".join(lines)


def write_report(report: EvalReport, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{stamp}-{report.suite}-{report.dataset}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path
