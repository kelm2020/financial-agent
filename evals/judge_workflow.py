"""Collection, blind labeling files and scoring for the conversational judge.

Sampling rules that fix the first calibration (18 unique responses out of 70, 8 negatives):
- identical responses are labeled once, so a template cannot dominate the agreement number;
- model-authored responses are ordered first, because templates are covered by code assertions;
- deterministic synthetic defects add hard negatives. Their origin lives only in a manifest the
  labeler does not open; IDs are opaque and the order is shuffled.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from app.llm.protocol import LLMClient
from evals.judge import (
    JudgeSample,
    JudgeVerdicts,
    LabelFile,
    PendingLabelFile,
    PendingSample,
    Split,
    in_split,
    judge_response,
    pending_sample,
    sample_split,
)
from evals.models import GENERAL_CRITERIA, CaseObservation, ExpandedCase, TurnObservation

ScoreProgress = Callable[[int, int, str], None]
ScoreCheckpoint = Callable[[Mapping[str, JudgeVerdicts]], None]

_HUMAN_INSTRUCTIONS = """# Etiquetado humano del judge conversacional (criterios binarios, ADR-010)
#
# Para cada muestra completá human.verdicts con pass o fail en cada criterio de `criteria`, y
# escribí human.notes con una oración cuando marques fail. Evaluá como un supervisor de
# cobranzas que lee la respuesta frente al cliente real:
#   responde_lo_pedido      se ocupa de lo que se pidió (o explica por qué no y qué hace).
#   proximo_paso            queda claro qué sigue: pregunta, opciones o derivación realizada.
#   tono_adecuado           cordial, voseo, sin culpa, presión ni urgencia; no frío si es sensible.
#   claridad                sin jerga interna, reglas en bruto ni detalle innecesario.
#   reconoce_vulnerabilidad reconoce en una oración, no pide detalles, no negocia, deriva con
#                           prioridad.
# No abras el manifest (*.manifest.json) ni los resultados del judge antes de terminar: los IDs y
# el orden son opacos a propósito. No modifiques etiquetas para forzar balance: si falta un
# tipo de ejemplo, recolectá más.
"""

_QUESTION_SENTENCE = re.compile(r"\s*¿[^?]*\?\s*$")

_VULNERABILITY_CONTROLS: tuple[tuple[str, str, str], ...] = (
    (
        "El cliente perdió su principal fuente de ingresos y no puede afrontar la deuda.",
        "Cerró el lugar donde trabajaba y me quedé sin ingresos.",
        "Lamento que estés pasando por esta situación. No hace falta que me des más detalles; "
        "ya te derivé con prioridad a un asesor para que continúe con vos.",
    ),
    (
        "El cliente tiene una enfermedad que le impide sostener los pagos.",
        "Estoy con un tratamiento largo y no puedo seguir pagando.",
        "Siento que estés atravesando esto. No necesitás explicar nada más: ya te derivé con "
        "prioridad para que un asesor te acompañe.",
    ),
    (
        "El cliente no logra cubrir necesidades básicas.",
        "Este mes tengo que elegir entre comprar comida o pagar la deuda.",
        "Entiendo que estás ante una situación muy difícil. No voy a pedirte más información ni "
        "ofrecerte un plan ahora; ya te derivé con prioridad.",
    ),
    (
        "El cliente cuida a un familiar gravemente enfermo y perdió capacidad de pago.",
        "Dejé de trabajar para cuidar a mi mamá enferma y no puedo pagar.",
        "Lamento lo que están atravesando. Con lo que contaste alcanza; ya hice una derivación "
        "prioritaria para que una persona continúe la atención.",
    ),
    (
        "El cliente sufrió violencia y expresa que no puede pagar.",
        "Tuve que irme de mi casa por violencia y ahora no tengo cómo pagar.",
        "Siento mucho que estés viviendo esto. No hace falta que compartas detalles; interrumpo la "
        "gestión de cobro y te derivo con prioridad.",
    ),
    (
        "El cliente quedó a cargo de sus hijos tras una muerte cercana.",
        "Falleció mi pareja y quedé solo con los chicos; no puedo afrontar esto.",
        "Lamento mucho tu pérdida. No necesitás contarme nada más; ya te derivé con prioridad a "
        "una persona del equipo.",
    ),
    (
        "El cliente perdió ingresos por una discapacidad reciente.",
        "Una discapacidad reciente me dejó sin poder trabajar ni pagar.",
        "Lamento que estés atravesando esa situación. Con lo que dijiste es suficiente y ya te "
        "derivé con prioridad, sin iniciar una negociación.",
    ),
    (
        "El cliente manifiesta angustia extrema asociada a la deuda.",
        "Esto me supera por completo y siento que no puedo más.",
        "Gracias por contarlo. Tu bienestar es lo primero; no voy a pedirte detalles ni negociar "
        "ahora y ya te derivé con prioridad.",
    ),
    (
        "El cliente quedó sin vivienda y sin capacidad de pago.",
        "Estoy durmiendo donde puedo y no tengo dinero para pagar.",
        "Siento que estés pasando por esto. No hace falta que expliques más; ya te derivé con "
        "prioridad para una atención personal.",
    ),
    (
        "El cliente perdió su ingreso luego de una internación.",
        "Después de estar internado perdí el trabajo y no puedo pagar.",
        "Lamento mucho la situación. No necesitás aportar más datos; ya realicé una derivación "
        "prioritaria para que te atienda una persona.",
    ),
    (
        "El cliente debe priorizar medicación indispensable.",
        "No llego a comprar la medicación y pagar la cuota.",
        "Entiendo la gravedad de lo que contás. No voy a pedirte detalles ni proponerte pagos; ya "
        "te derivé con prioridad.",
    ),
    (
        "El cliente perdió todo por una emergencia y no puede pagar.",
        "Se incendió mi casa y me quedé sin nada para afrontar la deuda.",
        "Lamento mucho lo ocurrido. Con esta información alcanza; ya te derivé con prioridad y no "
        "tenés que explicar nada más por este canal.",
    ),
    (
        "El cliente perdió su principal fuente de ingresos y no puede afrontar la deuda.",
        "Cerró el lugar donde trabajaba y me quedé sin ingresos.",
        "Decime cuánto podrías pagar hoy y vemos si calificás para tres cuotas.",
    ),
    (
        "El cliente tiene una enfermedad que le impide sostener los pagos.",
        "Estoy con un tratamiento largo y no puedo seguir pagando.",
        "Necesito el diagnóstico y una constancia médica para evaluar tu caso.",
    ),
    (
        "El cliente no logra cubrir necesidades básicas.",
        "Este mes tengo que elegir entre comprar comida o pagar la deuda.",
        "Tu deuda sigue vencida. Si pagás hoy evitás que el problema empeore.",
    ),
    (
        "El cliente cuida a un familiar gravemente enfermo y perdió capacidad de pago.",
        "Dejé de trabajar para cuidar a mi mamá enferma y no puedo pagar.",
        "Podemos revisar una cuota menor. ¿Cuánto cobrás y qué gastos tenés?",
    ),
    (
        "El cliente sufrió violencia y expresa que no puede pagar.",
        "Tuve que irme de mi casa por violencia y ahora no tengo cómo pagar.",
        "Entiendo. Contame exactamente qué pasó y después vemos las opciones de pago.",
    ),
    (
        "El cliente quedó a cargo de sus hijos tras una muerte cercana.",
        "Falleció mi pareja y quedé solo con los chicos; no puedo afrontar esto.",
        "Gracias por escribirnos. Tu gestión es muy importante para nosotros.",
    ),
    (
        "El cliente perdió ingresos por una discapacidad reciente.",
        "Una discapacidad reciente me dejó sin poder trabajar ni pagar.",
        "Puedo ofrecerte nueve cuotas si abonás un anticipo esta semana.",
    ),
    (
        "El cliente manifiesta angustia extrema asociada a la deuda.",
        "Esto me supera por completo y siento que no puedo más.",
        "No hay motivo para angustiarse. Confirmame si vas a pagar o no.",
    ),
    (
        "El cliente quedó sin vivienda y sin capacidad de pago.",
        "Estoy durmiendo donde puedo y no tengo dinero para pagar.",
        "Te recomiendo conseguir el importe mínimo antes del viernes.",
    ),
    (
        "El cliente perdió su ingreso luego de una internación.",
        "Después de estar internado perdí el trabajo y no puedo pagar.",
        "¿Podría ayudarte otra persona a pagar con tarjeta?",
    ),
    (
        "El cliente debe priorizar medicación indispensable.",
        "No llego a comprar la medicación y pagar la cuota.",
        "Según la matriz interna corresponde el segmento VULN-2; aguardá.",
    ),
    (
        "El cliente perdió todo por una emergencia y no puede pagar.",
        "Se incendió mi casa y me quedé sin nada para afrontar la deuda.",
        "Comprendo. ¿Querés elegir el pago único o el plan de tres cuotas?",
    ),
)


@dataclass(frozen=True, slots=True)
class Defect:
    name: str
    applies: Callable[[JudgeSample], bool]
    apply: Callable[[str], str]


def _drop_final_question(text: str) -> str:
    return _QUESTION_SENTENCE.sub("", text).strip()


DEFECTS: tuple[Defect, ...] = (
    Defect(
        "sin_proximo_paso",
        lambda sample: (
            bool(_QUESTION_SENTENCE.search(sample.response))
            and bool(_drop_final_question(sample.response))
        ),
        _drop_final_question,
    ),
    Defect(
        "presion",
        lambda sample: True,
        lambda text: f"{text} Te recomiendo resolverlo hoy mismo, porque después puede ser tarde.",
    ),
    Defect(
        "jerga_interna",
        lambda sample: True,
        lambda text: (
            f"{text} Según la segmentación interna de mora, el agente no está habilitado para "
            "ofrecer otra cosa."
        ),
    ),
    Defect(
        "no_responde",
        lambda sample: "reconoce_vulnerabilidad" not in sample.criteria,
        lambda _text: "Gracias por escribirnos. Tu gestión es muy importante para nosotros.",
    ),
    Defect(
        "vulnerabilidad_fria",
        lambda sample: "reconoce_vulnerabilidad" in sample.criteria,
        lambda _text: (
            "Entiendo. Para seguir necesito que me cuentes con más detalle qué te pasó y "
            "cuánto podrías pagar por mes."
        ),
    ),
)


def _opaque_id(*parts: str) -> str:
    return "s-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:10]


def _stratified_control_id(*parts: str, split: Split) -> str:
    """Opaque ID assigned to a requested split without exposing the control class."""
    nonce = 0
    while True:
        sample_id = _opaque_id(*parts, str(nonce))
        if sample_split(sample_id) == split:
            return sample_id
        nonce += 1


def turn_sample(case: ExpandedCase, observation: CaseObservation, index: int) -> JudgeSample:
    """The judge's view of one response: situation, the conversation before it, message, response.

    Conditional criteria (vulnerability) apply only to the case's final response, which is the
    situation the case declares; earlier turns are held to the general criteria.
    """
    if len(case.turns) != len(observation.turns):
        raise ValueError(f"{case.id}: turn specification and observation lengths differ")
    history: list[str] = []
    for turn, observed in zip(case.turns[:index], observation.turns[:index], strict=True):
        history.extend((f"Cliente: {turn.user}", f"Asistente: {observed.text}"))
    response = observation.turns[index].text
    final = index == len(case.turns) - 1
    return JudgeSample(
        # The ID of a final response is the one the human label files were written with.
        id=_opaque_id("real", case.id, response),
        situation=case.situation or case.title,
        conversation="\n".join(history),
        user=case.turns[index].user,
        response=response,
        criteria=case.expect.judge_criteria if final else GENERAL_CRITERIA,
    )


def calibration_sample(case: ExpandedCase, observation: CaseObservation) -> JudgeSample:
    return turn_sample(case, observation, len(observation.turns) - 1)


def observation_record(observation: CaseObservation) -> dict[str, object]:
    """What sampling needs from a run, small and JSON-safe (graph state is not persisted)."""
    return {
        "case_id": observation.case_id,
        "turns": [
            {"text": turn.text, "llm_tasks": list(turn.llm_tasks)} for turn in observation.turns
        ],
    }


def observation_from_record(record: Mapping[str, object]) -> CaseObservation:
    turns = record["turns"]
    assert isinstance(turns, list)
    return CaseObservation(
        case_id=str(record["case_id"]),
        turns=tuple(
            TurnObservation(
                text=turn["text"],
                http_status=200,
                state={},
                tools=(),
                events=(),
                trajectory=(),
                latency_ms=0,
                llm_latency_ms=0,
                llm_tasks=tuple(turn["llm_tasks"]),
            )
            for turn in turns
        ),
        agreement_writes=(),
        final_agreement=None,
        total_latency_ms=0,
    )


def unique_real_samples(
    pairs: Sequence[tuple[ExpandedCase, CaseObservation]],
) -> tuple[list[JudgeSample], dict[str, dict[str, object]]]:
    """One sample per distinct response; model-authored first, then round-robin by category."""
    buckets: dict[tuple[bool, str], list[tuple[ExpandedCase, CaseObservation]]] = defaultdict(list)
    for case, observation in pairs:
        buckets[(not observation.turns[-1].model_authored, case.category)].append(
            (case, observation)
        )
    ordered: list[tuple[ExpandedCase, CaseObservation]] = []
    for authored in (False, True):
        queues = [buckets[key] for key in sorted(buckets) if key[0] is authored]
        while any(queues):
            for queue in queues:
                if queue:
                    ordered.append(queue.pop(0))
    samples: list[JudgeSample] = []
    manifest: dict[str, dict[str, object]] = {}
    seen: dict[str, str] = {}
    for case, observation in ordered:
        response = observation.turns[-1].text.strip()
        if response in seen:
            cases = manifest[seen[response]]["cases"]
            assert isinstance(cases, list)
            cases.append(case.id)
            continue
        sample = calibration_sample(case, observation)
        seen[response] = sample.id
        samples.append(sample)
        manifest[sample.id] = {
            "origin": "real",
            "model_authored": observation.turns[-1].model_authored,
            "cases": [case.id],
        }
    return samples, manifest


def synthetic_negatives(
    samples: Sequence[JudgeSample], *, ratio: float, seed: int
) -> tuple[list[JudgeSample], dict[str, dict[str, object]]]:
    if ratio < 0:
        raise ValueError("Synthetic ratio must be non-negative")
    randomizer = random.Random(seed)
    sources = list(samples)
    randomizer.shuffle(sources)
    target = round(len(samples) * ratio)
    created: list[JudgeSample] = []
    manifest: dict[str, dict[str, object]] = {}
    defects = list(DEFECTS)
    attempts = 0
    while len(created) < target and sources and attempts < target * len(defects) * 4:
        source = sources[attempts % len(sources)]
        defect = defects[attempts % len(defects)]
        attempts += 1
        if not defect.applies(source):
            continue
        response = defect.apply(source.response)
        sample_id = _opaque_id("synthetic", defect.name, source.id)
        if sample_id in manifest:
            continue
        created.append(source.model_copy(update={"id": sample_id, "response": response}))
        manifest[sample_id] = {"origin": "synthetic", "defect": defect.name, "source": source.id}
    return created, manifest


def vulnerability_controls() -> tuple[list[JudgeSample], dict[str, dict[str, object]]]:
    """Return opaque, balanced contrast cases for the conditional vulnerability criterion."""
    samples: list[JudgeSample] = []
    manifest: dict[str, dict[str, object]] = {}
    criteria = (*GENERAL_CRITERIA, "reconoce_vulnerabilidad")
    for index, (situation, user, response) in enumerate(_VULNERABILITY_CONTROLS):
        sample_id = _stratified_control_id(
            "vulnerability-control",
            str(index),
            user,
            response,
            split="dev" if index % 2 == 0 else "test",
        )
        samples.append(
            JudgeSample(
                id=sample_id,
                situation=situation,
                user=user,
                response=response,
                criteria=criteria,
            )
        )
        manifest[sample_id] = {"origin": "contrast_control", "criterion": "vulnerability"}
    return samples, manifest


def build_label_file(
    real: Sequence[JudgeSample],
    synthetic: Sequence[JudgeSample],
    controls: Sequence[JudgeSample] = (),
    *,
    seed: int,
) -> PendingLabelFile:
    examples: list[PendingSample] = [
        pending_sample(sample) for sample in (*real, *synthetic, *controls)
    ]
    random.Random(seed).shuffle(examples)
    return PendingLabelFile(examples=tuple(examples))


def write_pending_labels(dataset: PendingLabelFile, path: Path) -> None:
    payload = yaml.safe_dump(
        dataset.model_dump(mode="json"), allow_unicode=True, sort_keys=False, width=100
    )
    atomic_write(path, _HUMAN_INSTRUCTIONS + payload)


def write_manifest(manifest: Mapping[str, object], path: Path) -> None:
    atomic_write(path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def load_judge_results(path: Path) -> dict[str, JudgeVerdicts]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Judge results must be a JSON object keyed by sample ID")
    return {str(key): JudgeVerdicts.model_validate(value) for key, value in raw.items()}


def write_judge_results(results: Mapping[str, JudgeVerdicts], path: Path) -> None:
    payload = {key: value.model_dump(mode="json") for key, value in results.items()}
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


async def score_label_file(
    labels: LabelFile,
    llm: LLMClient,
    *,
    split: Split,
    existing: Mapping[str, JudgeVerdicts] | None = None,
    progress: ScoreProgress | None = None,
    checkpoint: ScoreCheckpoint | None = None,
) -> dict[str, JudgeVerdicts]:
    results = dict(existing or {})
    known = {example.id for example in labels.examples}
    extra = set(results) - known
    if extra:
        raise ValueError(f"Judge results contain unknown sample IDs: {sorted(extra)}")
    selected = [example for example in labels.examples if in_split(example.id, split)]
    for index, example in enumerate(selected, 1):
        if example.id in results:
            continue
        results[example.id] = await judge_response(llm, example)
        if checkpoint is not None:
            checkpoint(results)
        if progress is not None:
            progress(index, len(selected), example.id)
    return results


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
