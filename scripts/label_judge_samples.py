"""Terminal labeler for the blind conversational-judge file (ADR-010).

Shows one sample at a time — situation, previous turns, the customer's message and the response —
and asks pass/fail per criterion. Every answered sample is saved atomically, so the session can
stop at any point and resume later. It never reads the manifest: origin stays hidden.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from evals.judge import PendingSample, load_pending_labels
from evals.judge_workflow import write_pending_labels

DEFAULT_DATASET = Path("evals/judge_calibration.yaml")

CRITERIA_HELP = {
    "responde_lo_pedido": (
        "pass si se ocupa de lo que el cliente pidió o explica por qué no y qué hace en su lugar."
    ),
    "proximo_paso": "pass si queda claro qué sigue: una pregunta, opciones o una derivación hecha.",
    "tono_adecuado": (
        "pass si es cordial y sin culpa, presión ni urgencia; fail si es frío ante algo sensible."
    ),
    "claridad": "pass si se entiende sin jerga interna, reglas en bruto ni detalle innecesario.",
    "reconoce_vulnerabilidad": (
        "pass si reconoce lo contado en una oración, no pide detalles, no negocia y deriva con "
        "prioridad."
    ),
}

type Reader = Callable[[str], str]
type Writer = Callable[[str], None]


def _show(write: Writer, position: int, total: int, example: PendingSample) -> None:
    write("")
    write(f"── Muestra {position}/{total} · {example.id} " + "─" * 30)
    write(f"Situación: {example.situation}")
    write(f"Conversación previa:\n{example.conversation or '(sin turnos previos)'}")
    write(f"Cliente: {example.user}")
    write(f"Respuesta a evaluar:\n{example.response}")


def label(path: Path, *, read: Reader = input, write: Writer = print) -> int:
    """Label pending samples interactively. Returns how many samples remain incomplete."""
    dataset = load_pending_labels(path)
    pending = [
        example
        for example in dataset.examples
        if any(value is None for value in example.human.verdicts.values())
    ]
    write(f"{len(dataset.examples) - len(pending)}/{len(dataset.examples)} muestras completas.")
    write("Respuestas: p = pass · f = fail · ? = ayuda · s = saltar muestra · q = guardar y salir")
    for position, example in enumerate(pending, 1):
        _show(write, position, len(pending), example)
        skipped = False
        for criterion in example.criteria:
            if example.human.verdicts.get(criterion) is not None:
                continue
            while True:
                answer = read(f"  {criterion} [p/f/?/s/q]: ").strip().casefold()
                if answer == "?":
                    write(f"    {CRITERIA_HELP[criterion]}")
                elif answer == "q":
                    write_pending_labels(dataset, path)
                    return _remaining(path)
                elif answer == "s":
                    skipped = True
                    break
                elif answer in {"p", "f"}:
                    example.human.verdicts[criterion] = "pass" if answer == "p" else "fail"
                    break
                else:
                    write("    Respondé p, f, ?, s o q.")
            if skipped:
                break
        has_fail = "fail" in example.human.verdicts.values()
        if not skipped and has_fail and not example.human.notes.strip():
            example.human.notes = read("  Nota breve sobre el fail: ").strip()
        write_pending_labels(dataset, path)
    remaining = _remaining(path)
    write(f"Listo. Muestras incompletas: {remaining}.")
    return remaining


def _remaining(path: Path) -> int:
    return sum(
        any(value is None for value in example.human.verdicts.values())
        for example in load_pending_labels(path).examples
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label the blind judge file in the terminal")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    label(parse_args(argv).dataset)


if __name__ == "__main__":  # pragma: no cover
    main()
