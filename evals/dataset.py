from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]

from evals.models import CaseFile, CaseSpec

ROOT = Path(__file__).parent
# Development set: the 22 scenarios of §11.3 plus cases promoted from the blind suite. It is the
# only suite allowed to shape lexicons and templates.
CASES_DIR = ROOT / "cases"
# Engineer-written paraphrases and hard cases (regression, not blind).
HELDOUT_DIR = ROOT / "heldout"
# Phrasings written by a model that never saw the router or templates (scripts/generate_blind_
# phrasings.py). Language understanding is the model's job, so level A gates only on safety.
BLIND_DIR = ROOT / "blind"
CANONICAL_BASE_CASES = 42  # 22 from §11.3 + 9 promoted + 11 local chat regressions

type DatasetName = Literal["canonical", "heldout", "blind"]
DATASETS: dict[DatasetName, tuple[Path, int | None]] = {
    "canonical": (CASES_DIR, CANONICAL_BASE_CASES),
    "heldout": (HELDOUT_DIR, None),
    "blind": (BLIND_DIR, None),
}


def load_cases(
    path: Path = CASES_DIR, *, expected_base: int | None = CANONICAL_BASE_CASES
) -> tuple[CaseSpec, ...]:
    cases: list[CaseSpec] = []
    for source in sorted(path.glob("*.yaml")):
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        cases.extend(CaseFile.model_validate(payload).cases)
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Base evaluation case IDs must be unique")
    if expected_base is not None and len(cases) != expected_base:
        raise ValueError(f"Expected {expected_base} base cases in {path}, found {len(cases)}")
    return tuple(cases)


def load_dataset(name: DatasetName) -> tuple[CaseSpec, ...]:
    path, expected = DATASETS[name]
    return load_cases(path, expected_base=expected)
