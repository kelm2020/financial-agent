from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from evals.dataset import load_cases
from evals.judge import JudgeSample, PendingLabelFile, load_pending_labels, pending_sample
from evals.judge_workflow import write_pending_labels
from scripts import generate_blind_phrasings as blind_script
from scripts import label_judge_samples as label_script
from tests.agent_support import offline_settings

# ---------------------------------------------------------------------------------- labeler


def _labels(path: Path) -> None:
    samples = (
        JudgeSample(
            id="s-aaa", situation="Consulta de saldo.", user="¿Cuánto debo?", response="r1"
        ),
        JudgeSample(
            id="s-bbb",
            situation="Vulnerabilidad.",
            conversation="Cliente: hola\nAsistente: hola",
            user="Perdí el trabajo",
            response="r2",
            criteria=(
                "responde_lo_pedido",
                "proximo_paso",
                "tono_adecuado",
                "claridad",
                "reconoce_vulnerabilidad",
            ),
        ),
    )
    write_pending_labels(PendingLabelFile(examples=tuple(map(pending_sample, samples))), path)


def _reader(answers: Sequence[str]) -> tuple[list[str], label_script.Reader]:
    prompts: list[str] = []
    iterator: Iterator[str] = iter(answers)

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return next(iterator)

    return prompts, read


def test_terminal_labeler_saves_every_sample_and_resumes(tmp_path: Path) -> None:
    path = tmp_path / "labels.yaml"
    _labels(path)
    output: list[str] = []

    _, read = _reader(["?", "x", "p", "f", "p", "p", "nota breve", "s"])
    assert label_script.label(path, read=read, write=output.append) == 1
    first = load_pending_labels(path).examples[0].human
    assert first.verdicts == {
        "responde_lo_pedido": "pass",
        "proximo_paso": "fail",
        "tono_adecuado": "pass",
        "claridad": "pass",
    }
    assert first.notes == "nota breve"
    joined = "\n".join(output)
    assert "pass si queda claro" not in joined  # help was asked for the first criterion only
    assert "pass si se ocupa" in joined and "Respondé p, f, ?, s o q." in joined
    assert "Situación: Consulta de saldo." in joined and "(sin turnos previos)" in joined
    assert "Muestras incompletas: 1." in joined

    _, read = _reader(["p", "q"])
    assert label_script.label(path, read=read, write=output.append) == 1
    assert load_pending_labels(path).examples[1].human.verdicts["responde_lo_pedido"] == "pass"

    prompts, read = _reader(["p", "p", "p", "p"])
    assert label_script.label(path, read=read, write=output.append) == 0
    assert all("responde_lo_pedido" not in prompt for prompt in prompts)  # answered ones skipped
    assert load_pending_labels(path).examples[1].human.notes == ""


def test_labeler_main(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []

    def fake_label(path: Path) -> int:
        seen.append(path)
        return 0

    monkeypatch.setattr(label_script, "label", fake_label)
    label_script.main(["--dataset", "otro.yaml"])
    assert seen == [Path("otro.yaml")]


# ---------------------------------------------------------------------- blind phrasings


class _Writer:
    instances: ClassVar[list[_Writer]] = []
    duplicates_only: ClassVar[bool] = False

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls = 0
        self.closed = False
        type(self).instances.append(self)

    async def complete[T: BaseModel](
        self,
        *,
        task: str,
        messages: Sequence[Mapping[str, str]],
        response_model: type[T],
    ) -> T:
        assert task == "blind_phrasings"
        prompt = messages[1]["content"]
        assert "routing" not in prompt and "plantilla" not in prompt
        self.calls += 1
        if type(self).duplicates_only:
            phrases = ["Quiero hablar con un asesor", " "]
        else:
            phrases = [
                "Quiero hablar con un asesor",
                "",
                *(f"frase {self.calls}-{n}" for n in range(4)),
            ]
        return response_model.model_validate({"phrases": phrases})

    async def aclose(self) -> None:
        self.closed = True


async def test_blind_phrasings_are_new_loadable_and_never_regenerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "heldout" / "blind.yaml"
    _Writer.instances = []
    _Writer.duplicates_only = False
    monkeypatch.setattr(blind_script, "OpenAIResponsesLLM", _Writer)
    argv = ["--output", str(output), "--per-category", "2"]

    monkeypatch.setattr(blind_script, "get_settings", lambda: offline_settings())
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await blind_script.run(blind_script.parse_args(argv))
    monkeypatch.setattr(blind_script, "get_settings", lambda: offline_settings(openai_api_key="t"))
    with pytest.raises(RuntimeError, match="OPENAI_JUDGE_MODEL"):
        await blind_script.run(blind_script.parse_args(argv))
    with pytest.raises(ValueError, match="must differ"):
        await blind_script.run(blind_script.parse_args([*argv, "--model", "gpt-5-nano"]))

    assert await blind_script.run(blind_script.parse_args([*argv, "--model", "writer"])) == output
    content = output.read_text(encoding="utf-8")
    assert "# Modelo: writer · prompt: blind_phrasings" in content
    assert "Quiero hablar con un asesor" not in content  # existing phrasings are filtered out
    cases = load_cases(output.parent, expected_base=None)
    assert len(cases) == len(blind_script.CATEGORIES)
    assert all(len(case.variant_axes[0].values) == 2 for case in cases)
    confirmation = next(case for case in cases if case.id == "A-61")
    assert confirmation.turns[0].user == "Quiero la opción de 3 cuotas"
    vulnerable = next(case for case in cases if case.id == "E-61")
    assert vulnerable.expect.judge_rubric == ("reconoce_vulnerabilidad",)
    (writer,) = _Writer.instances
    assert writer.closed and writer.kwargs["max_output_tokens"] == 4000

    with pytest.raises(FileExistsError, match="never regenerated"):
        await blind_script.run(blind_script.parse_args([*argv, "--model", "writer"]))

    output.unlink()
    _Writer.duplicates_only = True
    with pytest.raises(ValueError, match="new phrasings"):
        await blind_script.run(blind_script.parse_args([*argv, "--model", "writer"]))
    assert not output.exists()


def test_blind_phrasings_main(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    async def fake_run(args: object) -> Path:
        seen.append(args.per_category)  # type: ignore[attr-defined]
        return Path("blind.yaml")

    monkeypatch.setattr(blind_script, "run", fake_run)
    blind_script.main(["--per-category", "3"])
    assert seen == [3]


async def test_promoted_blind_phrasings_are_replaced_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "blind" / "cases.yaml"
    _Writer.instances = []
    _Writer.duplicates_only = False
    monkeypatch.setattr(blind_script, "OpenAIResponsesLLM", _Writer)
    monkeypatch.setattr(blind_script, "get_settings", lambda: offline_settings(openai_api_key="t"))
    base = ["--output", str(output), "--per-category", "2", "--model", "writer"]
    await blind_script.run(blind_script.parse_args(base))
    before = load_cases(output.parent, expected_base=None)
    old = next(case for case in before if case.id == "A-61")

    await blind_script.run(blind_script.parse_args([*base, "--replace", "A-61:b1"]))
    content = output.read_text(encoding="utf-8")
    assert "# Reemplazada A-61:b1 (writer," in content
    assert content.startswith("# Frases held-out CIEGAS")
    after = next(
        case for case in load_cases(output.parent, expected_base=None) if case.id == "A-61"
    )
    new_phrase = after.variant_axes[0].values[0].turn_text[1]
    assert new_phrase != old.variant_axes[0].values[0].turn_text[1]
    assert after.turns[1].user == new_phrase  # b1 also defines the base case
    assert after.variant_axes[0].values[1] == old.variant_axes[0].values[1]

    for target, message in (("Z-99:b1", "Unknown blind case"), ("A-61:b9", "Unknown variant")):
        with pytest.raises(ValueError, match=message):
            await blind_script.run(blind_script.parse_args([*base, "--replace", target]))
    _Writer.duplicates_only = True
    with pytest.raises(ValueError, match="no new phrasing"):
        await blind_script.run(blind_script.parse_args([*base, "--replace", "A-61:b2"]))
