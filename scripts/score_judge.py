from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.llm.openai_responses import OpenAIResponsesLLM
from config.settings import get_settings
from evals.judge import JudgeVerdicts, in_split, load_labels
from evals.judge_workflow import load_judge_results, score_label_file, write_judge_results

DEFAULT_DATASET = Path("evals/judge_calibration.yaml")
DEFAULT_OUTPUT = Path("evals/judge_results.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a completed human labeling file with the conversational judge"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        default="dev",
        help="Iterate the judge prompt on dev; score test only to publish the agreement.",
    )
    parser.add_argument("--model")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _existing_results(path: Path, *, resume: bool) -> dict[str, JudgeVerdicts]:
    if not path.exists():
        return {}
    if not resume:
        raise FileExistsError(
            f"{path} already exists; use --resume to continue or add another split"
        )
    return load_judge_results(path)


async def score(args: argparse.Namespace) -> Path:
    dataset_path: Path = args.dataset
    output: Path = args.output
    labels = load_labels(dataset_path)
    existing = _existing_results(output, resume=args.resume)
    unknown = set(existing) - {example.id for example in labels.examples}
    if unknown:
        raise ValueError(f"Judge results contain unknown sample IDs: {sorted(unknown)}")
    selected = [example.id for example in labels.examples if in_split(example.id, args.split)]
    if all(sample_id in existing for sample_id in selected):
        print(f"Judge ({args.split}): {len(selected)}/{len(selected)} · Archivo: {output}")
        return output
    settings = get_settings()
    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required to score judge samples")
    model = args.model or settings.openai_judge_model
    if not model:
        raise RuntimeError("Set OPENAI_JUDGE_MODEL or JUDGE_MODEL before judge scoring")
    if model == settings.openai_agent_model:
        raise ValueError("The judge model must differ from OPENAI_AGENT_MODEL")
    llm = OpenAIResponsesLLM(
        api_key=key,
        model=model,
        max_output_tokens=8000,
        timeout_seconds=120,
        reasoning_effort="minimal" if model.startswith("gpt-5") else None,
    )

    def checkpoint(results: Mapping[str, JudgeVerdicts]) -> None:
        write_judge_results(results, output)

    try:
        results = await score_label_file(
            labels,
            llm,
            split=args.split,
            existing=existing,
            checkpoint=checkpoint,
            progress=lambda index, total, sample_id: print(
                f"[{index}/{total}] {sample_id}: puntuado", flush=True
            ),
        )
    finally:
        await llm.aclose()
    write_judge_results(results, output)
    print(f"Judge ({args.split}): {len(selected)} muestras · Modelo: {model} · Archivo: {output}")
    return output


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(score(parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover
    main()
