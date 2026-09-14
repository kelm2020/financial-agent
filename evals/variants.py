from __future__ import annotations

from itertools import product

from evals.models import CaseSpec, ExpandedCase, ExpectPatch, ExpectSpec, VariantValue


def _patched_expect(base: ExpectSpec, patch: ExpectPatch | None) -> ExpectSpec:
    if patch is None:
        return base
    values = {key: value for key, value in patch.model_dump().items() if value is not None}
    return base.model_copy(update=values)


def _apply_values(case: CaseSpec, values: tuple[VariantValue, ...]) -> ExpandedCase:
    turns = list(case.turns)
    expectation = case.expect
    situation = case.situation
    drop: set[int] = set()
    suffixes: list[str] = []
    for value in values:
        suffixes.append(value.id)
        for index, text in value.turn_text.items():
            if index < 0 or index >= len(turns):
                raise ValueError(f"{case.id}: variant turn index {index} is out of range")
            turns[index] = turns[index].model_copy(update={"user": text})
        for index, seconds in value.advance_before.items():
            if index < 0 or index >= len(turns):
                raise ValueError(f"{case.id}: advance turn index {index} is out of range")
            turns[index] = turns[index].model_copy(update={"advance_seconds": seconds})
        drop.update(value.drop_turns)
        situation = value.situation or situation
        expectation = _patched_expect(expectation, value.expect)
    turns = [turn for index, turn in enumerate(turns) if index not in drop]
    if not turns:
        raise ValueError(f"{case.id}: variants removed every turn")
    variant_id = "__".join(suffixes)
    return ExpandedCase(
        id=f"{case.id}__{variant_id}" if variant_id else case.id,
        base_id=case.id,
        title=case.title,
        category=case.category,
        customer_id=case.customer_id,
        situation=situation or case.title,
        turns=tuple(turns),
        evidence=case.evidence,
        setup=case.setup,
        expect=expectation,
    )


def expand_case(case: CaseSpec) -> tuple[ExpandedCase, ...]:
    if not case.variant_axes:
        return (_apply_values(case, ()),)
    combinations = product(*(axis.values for axis in case.variant_axes))
    return tuple(_apply_values(case, tuple(values)) for values in combinations)


def expand_cases(cases: tuple[CaseSpec, ...]) -> tuple[ExpandedCase, ...]:
    expanded = tuple(item for case in cases for item in expand_case(case))
    ids = [case.id for case in expanded]
    if len(ids) != len(set(ids)):
        raise ValueError("Expanded evaluation case IDs must be unique")
    return expanded
