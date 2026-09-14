"""Binary, per-criterion conversational judge and its calibration against human labels.

Design (ADR-010, replaces an earlier 0-2 Likert average):
- one pass/fail verdict per criterion, because adjacent Likert points are not reproducible
  between annotators and an average hides which failure mode moved;
- agreement is published per criterion as TPR/TNR (and Cohen's kappa), on a held-out ``test``
  split that is never used to edit the prompt;
- a criterion cannot be calibrated without enough human ``pass`` AND ``fail`` examples.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.llm.protocol import LLMClient
from app.prompts import JUDGE_PROMPT_PATH
from evals.models import ALL_CRITERIA, GENERAL_CRITERIA, JudgeCriterion

PROMPT_PATH = JUDGE_PROMPT_PATH
MINIMUM_SAMPLES = 40

type Verdict = Literal["pass", "fail"]
type Split = Literal["dev", "test", "all"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CriterionVerdict(_Strict):
    verdict: Literal["pass", "fail", "na"]
    reason: str


class JudgeVerdicts(_Strict):
    """Structured model output. Every field is required so the provider schema stays strict."""

    responde_lo_pedido: CriterionVerdict
    proximo_paso: CriterionVerdict
    tono_adecuado: CriterionVerdict
    claridad: CriterionVerdict
    reconoce_vulnerabilidad: CriterionVerdict

    def verdict(self, criterion: JudgeCriterion) -> Verdict:
        value = getattr(self, criterion).verdict
        # "na" on a criterion that applies is a judge error, never a free pass.
        return "pass" if value == "pass" else "fail"

    def acceptable(self, criteria: tuple[JudgeCriterion, ...]) -> bool:
        return all(self.verdict(criterion) == "pass" for criterion in criteria)


class JudgeSample(_Strict):
    id: str
    situation: str
    conversation: str = ""
    user: str
    response: str
    criteria: tuple[JudgeCriterion, ...] = GENERAL_CRITERIA

    @model_validator(mode="after")
    def general_criteria_always_apply(self) -> JudgeSample:
        missing = [item for item in GENERAL_CRITERIA if item not in self.criteria]
        if missing:
            raise ValueError(f"{self.id}: general criteria are mandatory, missing {missing}")
        return self


class PendingLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: dict[JudgeCriterion, Verdict | None]
    notes: str = ""


class PendingSample(JudgeSample):
    model_config = ConfigDict(extra="forbid", frozen=False)

    human: PendingLabel


class PendingLabelFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    examples: tuple[PendingSample, ...]

    @model_validator(mode="after")
    def unique_ids(self) -> PendingLabelFile:
        ids = [example.id for example in self.examples]
        if len(ids) != len(set(ids)):
            raise ValueError("Calibration sample IDs must be unique")
        return self


class HumanLabel(_Strict):
    verdicts: dict[JudgeCriterion, Verdict]
    notes: str = ""


class LabeledSample(JudgeSample):
    human: HumanLabel


class LabelFile(_Strict):
    examples: tuple[LabeledSample, ...]


class CriterionAgreement(_Strict):
    criterion: str
    samples: int = Field(ge=0)
    human_pass: int = Field(ge=0)
    human_fail: int = Field(ge=0)
    agreement: float = Field(ge=0, le=1)
    tpr: float | None = None
    tnr: float | None = None
    cohens_kappa: float | None = None


class JudgeCalibration(_Strict):
    split: Split
    samples: int
    criteria: tuple[CriterionAgreement, ...]
    overall: CriterionAgreement


def pending_sample(sample: JudgeSample) -> PendingSample:
    return PendingSample(
        **sample.model_dump(),
        human=PendingLabel(verdicts=dict.fromkeys(sample.criteria)),
    )


def sample_split(sample_id: str) -> Literal["dev", "test"]:
    """Stable 50/50 split by ID: iterate the prompt on dev, publish agreement on test."""
    return "dev" if hashlib.sha256(sample_id.encode()).digest()[0] % 2 == 0 else "test"


def in_split(sample_id: str, split: Split) -> bool:
    return split == "all" or sample_split(sample_id) == split


def load_pending_labels(path: Path) -> PendingLabelFile:
    return PendingLabelFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def require_human_labels(
    pending: PendingLabelFile, *, minimum_samples: int = MINIMUM_SAMPLES
) -> LabelFile:
    if len(pending.examples) < minimum_samples:
        raise ValueError(
            f"Judge calibration requires at least {minimum_samples} labeled samples; "
            f"found {len(pending.examples)}"
        )
    problems: list[str] = []
    for example in pending.examples:
        keys = set(example.human.verdicts)
        if keys != set(example.criteria):
            problems.append(f"{example.id}: labels must cover exactly {list(example.criteria)}")
        elif any(value is None for value in example.human.verdicts.values()):
            problems.append(example.id)
    if problems:
        preview = ", ".join(problems[:5])
        suffix = "..." if len(problems) > 5 else ""
        raise ValueError(f"Complete every human verdict (pass/fail); pending={preview}{suffix}")
    return LabelFile.model_validate(pending.model_dump())


def load_labels(path: Path, *, minimum_samples: int = MINIMUM_SAMPLES) -> LabelFile:
    return require_human_labels(load_pending_labels(path), minimum_samples=minimum_samples)


def _agreement(name: str, pairs: list[tuple[bool, bool]]) -> CriterionAgreement:
    """``pairs`` are (human_pass, judge_pass). Positive class = pass."""
    total = len(pairs)
    human_pass = sum(human for human, _ in pairs)
    human_fail = total - human_pass
    matches = sum(human == judge for human, judge in pairs)
    true_pass = sum(human and judge for human, judge in pairs)
    true_fail = sum(not human and not judge for human, judge in pairs)
    accuracy = matches / total if total else 0.0
    kappa: float | None = None
    if total:
        judge_pass = sum(judge for _, judge in pairs) / total
        observed_pass = human_pass / total
        expected = observed_pass * judge_pass + (1 - observed_pass) * (1 - judge_pass)
        kappa = (accuracy - expected) / (1 - expected) if expected < 1 else None
    return CriterionAgreement(
        criterion=name,
        samples=total,
        human_pass=human_pass,
        human_fail=human_fail,
        agreement=accuracy,
        tpr=true_pass / human_pass if human_pass else None,
        tnr=true_fail / human_fail if human_fail else None,
        cohens_kappa=kappa,
    )


def calibrate_judge(
    labels: LabelFile,
    results: Mapping[str, JudgeVerdicts],
    *,
    split: Split = "test",
    min_per_class: int = 5,
) -> JudgeCalibration:
    examples = [example for example in labels.examples if in_split(example.id, split)]
    missing = sorted(example.id for example in examples if example.id not in results)
    if missing:
        raise ValueError(f"Incomplete judge results for split {split!r}; missing={missing}")
    per_criterion: list[CriterionAgreement] = []
    short: list[str] = []
    for criterion in ALL_CRITERIA:
        pairs = [
            (
                example.human.verdicts[criterion] == "pass",
                results[example.id].verdict(criterion) == "pass",
            )
            for example in examples
            if criterion in example.criteria
        ]
        if not pairs:
            continue
        agreement = _agreement(criterion, pairs)
        if agreement.human_pass < min_per_class or agreement.human_fail < min_per_class:
            short.append(f"{criterion} (pass={agreement.human_pass}, fail={agreement.human_fail})")
        per_criterion.append(agreement)
    if short:
        raise ValueError(
            f"Not enough human pass AND fail labels in split {split!r} "
            f"(minimum {min_per_class} each): {', '.join(short)}. Collect harder cases or "
            "synthetic negatives instead of editing labels."
        )
    overall = _agreement(
        "aceptable",
        [
            (
                all(value == "pass" for value in example.human.verdicts.values()),
                results[example.id].acceptable(example.criteria),
            )
            for example in examples
        ],
    )
    return JudgeCalibration(
        split=split, samples=len(examples), criteria=tuple(per_criterion), overall=overall
    )


def judge_messages(sample: JudgeSample) -> tuple[dict[str, str], ...]:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    conversation = sample.conversation or "sin turnos previos"
    return (
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"Situación: {sample.situation}\n\n"
                f"Conversación previa:\n{conversation}\n\n"
                f"Mensaje del cliente:\n{sample.user}\n\n"
                f"Respuesta a evaluar:\n{sample.response}\n\n"
                f"Criterios a evaluar: {', '.join(sample.criteria)}"
            ),
        },
    )


async def judge_response(llm: LLMClient, sample: JudgeSample) -> JudgeVerdicts:
    return await llm.complete(
        task="judge", messages=judge_messages(sample), response_model=JudgeVerdicts
    )
