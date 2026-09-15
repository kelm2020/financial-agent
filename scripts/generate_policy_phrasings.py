"""Blind policy questions (§11.3, ADR-011).

The writer is a model different from the agent's. It receives only a business situation per
question family, never app/graph/ontology.py, app/graph/routing.py, the knowledge base, the section
labels or the expected answers, so the phrasings cannot be shaped by what they evaluate. The labels
(source, answerability, sections) belong to the situation and are written from the knowledge base.

Anti-contamination rule: phrasings are never edited to pass. A phrasing that motivates a change to
a lexicon, the ontology or the policy prompt moves to evals/policy_regression.yaml, and a new blind
phrasing replaces it here.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from app.guards.normalize import detection_skeleton
from app.llm.openai_responses import OpenAIResponsesLLM
from app.llm.protocol import LLMClient
from scripts.generate_blind_phrasings import (
    _SYSTEM,
    Phrasings,
    _refuse_existing,
    _write,
    _writer,
)

DEFAULT_OUTPUT = Path("evals/policy_blind.yaml")
KNOWN_QUESTION_FILES = (
    Path("evals/policy_regression.yaml"),
    Path("evals/policy_challenge.yaml"),
)
PROMPT_VERSION = "policy_phrasings"
EFFECTIVE_ON = "2026-09-12"


@dataclass(frozen=True, slots=True)
class Situation:
    case_id: str
    brief: str
    answerable: bool
    sections: tuple[str, ...]
    required: str
    source: str = "rag"


SITUATIONS: tuple[Situation, ...] = (
    Situation(
        "S-61",
        "El cliente quiere saber si le reducen algo de lo que debe si paga todo junto, de una vez.",
        True,
        ("POL-NEG-003",),
        "Reducción sólo de intereses y sólo en un pago único, según el segmento.",
    ),
    Situation(
        "S-62",
        "El cliente quiere saber si para pagar en cuotas tiene que poner plata al principio.",
        True,
        ("POL-NEG-005",),
        "Entrega inicial desde cierta cantidad de cuotas y porcentaje según la mora.",
    ),
    Situation(
        "S-63",
        "El cliente quiere saber si puede pagar sólo una parte de lo que debe, sin armar un plan.",
        True,
        ("FAQ-001", "POL-NEG-006"),
        "Pago parcial desde un mínimo del saldo; no reemplaza un acuerdo.",
    ),
    Situation(
        "S-64",
        "El cliente quiere mover la fecha en la que le vence una cuota.",
        True,
        ("FAQ-003",),
        "El canal automático no cambia fechas; lo evalúa un operador con anticipación.",
    ),
    Situation(
        "S-65",
        "El cliente quiere saber cuánto tarda en verse reflejado un pago según cómo lo haga.",
        True,
        ("PAY-MET-002",),
        "Plazos de acreditación por medio de pago.",
    ),
    Situation(
        "S-66",
        "El cliente pregunta si puede pagar con un medio poco común, como criptomonedas o un "
        "cheque de otra persona.",
        True,
        ("PAY-MET-003",),
        "Medios no habilitados.",
    ),
    Situation(
        "S-67",
        "El cliente pregunta qué le pasa si arma un plan de pagos y después deja de pagar una "
        "cuota.",
        True,
        ("POL-NEG-008", "FAQ-004"),
        "Caída del plan por una cuota impaga y sus consecuencias.",
    ),
    Situation(
        "S-68",
        "El cliente pregunta hasta cuándo vale una propuesta de pago que le hicieron.",
        True,
        ("POL-NEG-007", "FAQ-012"),
        "Vigencia general de las ofertas, sin inventar una fecha particular.",
    ),
    Situation(
        "N-61",
        "El cliente pregunta si lo que pague de la deuda le sirve para pagar menos impuestos.",
        False,
        (),
        "Abstenerse: no hay política impositiva.",
    ),
    Situation(
        "N-62",
        "El cliente pregunta la tasa de interés anual o el costo financiero total de financiar la "
        "deuda.",
        False,
        (),
        "Abstenerse: no hay tasa anual documentada.",
    ),
    Situation(
        "N-63",
        "El cliente pregunta cuánto le cobran de comisión por pagar con una transferencia.",
        False,
        (),
        "Abstenerse: no hay comisiones documentadas.",
    ),
    Situation(
        "N-64",
        "El cliente pregunta si puede pagar la deuda en dólares u otra moneda extranjera.",
        False,
        (),
        "Abstenerse: no se documenta pago en moneda extranjera.",
    ),
)


def _messages(
    situation: Situation, count: int, existing: Sequence[str]
) -> tuple[dict[str, str], ...]:
    avoid = "\n".join(f"- {text}" for text in existing) or "- (ninguna)"
    return (
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Situación a expresar como pregunta del cliente: {situation.brief}\n\n"
                f"Preguntas que ya existen y no hay que repetir:\n{avoid}\n\n"
                f"Devolvé exactamente {count} preguntas distintas del cliente, una por elemento."
            ),
        },
    )


def known_questions(paths: Sequence[Path] = KNOWN_QUESTION_FILES) -> list[str]:
    questions: list[str] = []
    for path in paths:
        if path.exists():
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            questions.extend(str(case["query"]) for case in payload["cases"])
    return questions


async def generate(
    llm: LLMClient, *, per_situation: int, existing: Sequence[str]
) -> dict[str, list[str]]:
    seen = {detection_skeleton(text) for text in existing}
    result: dict[str, list[str]] = {}
    for situation in SITUATIONS:
        reply = await llm.complete(
            task="policy_phrasings",
            messages=_messages(situation, per_situation + 2, existing),
            response_model=Phrasings,
        )
        unique: list[str] = []
        for phrase in reply.phrases:
            text = phrase.strip()
            if text and detection_skeleton(text) not in seen:
                seen.add(detection_skeleton(text))
                unique.append(text)
        if len(unique) < per_situation:
            raise ValueError(
                f"{situation.case_id}: the writer returned {len(unique)} new questions, "
                f"{per_situation} required"
            )
        result[situation.case_id] = unique[:per_situation]
    return result


def case_file(phrasings: dict[str, list[str]]) -> dict[str, Any]:
    """The format of scripts/evaluate_policy_pipeline.py: labels from the situation, text blind."""
    cases = [
        {
            "id": f"{situation.case_id}:b{number}",
            "query": query,
            "source": situation.source,
            "answerable": situation.answerable,
            "sections": list(situation.sections),
            "required": situation.required,
        }
        for situation in SITUATIONS
        for number, query in enumerate(phrasings[situation.case_id], 1)
    ]
    return {"effective_on": EFFECTIVE_ON, "cases": cases}


def _briefs_digest() -> str:
    raw = json.dumps([situation.brief for situation in SITUATIONS], ensure_ascii=False)
    return hashlib.sha256((_SYSTEM + raw).encode()).hexdigest()[:12]


def render(payload: dict[str, Any], *, model: str, generated_at: datetime) -> str:
    header = (
        "# Preguntas de política CIEGAS (ADR-011). Las escribió un modelo distinto del agente,\n"
        "# que sólo recibió la situación de negocio: nunca vio la ontología, el router, la base\n"
        "# de conocimiento, las secciones ni las respuestas esperadas. Las etiquetas son de la\n"
        "# situación.\n"
        f"# Modelo: {model} · prompt: {PROMPT_VERSION} ({_briefs_digest()}) · "
        f"fecha: {generated_at:%Y-%m-%d}\n"
        "# No editar preguntas para que pasen. Si una motiva un cambio de léxico, ontología o\n"
        "# prompt, pasa a evals/policy_regression.yaml y se genera otra para este archivo.\n"
    )
    return header + str(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=100))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate blind policy questions")
    parser.add_argument("--model", help="Writer model; defaults to OPENAI_JUDGE_MODEL")
    parser.add_argument("--per-situation", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> Path:
    output: Path = args.output
    _refuse_existing(output)
    key, model = _writer(args.model)
    llm = OpenAIResponsesLLM(api_key=key, model=model, max_output_tokens=4000, timeout_seconds=120)
    try:
        phrasings = await generate(
            llm, per_situation=args.per_situation, existing=known_questions()
        )
    finally:
        await llm.aclose()
    payload: dict[str, Any] = case_file(phrasings)
    _write(output, render(payload, model=model, generated_at=datetime.now(UTC)))
    print(f"Preguntas ciegas: {len(payload['cases'])} · Archivo: {output}")
    return output


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover
    main()
