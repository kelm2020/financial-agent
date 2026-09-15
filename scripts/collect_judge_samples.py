from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from app.llm.openai_responses import build_agent_llm
from config.settings import get_settings
from evals.dataset import DATASETS, load_dataset
from evals.environment import run_case
from evals.judge_workflow import (
    build_label_file,
    observation_from_record,
    observation_record,
    synthetic_negatives,
    unique_real_samples,
    vulnerability_controls,
    write_manifest,
    write_pending_labels,
)
from evals.models import CaseObservation
from evals.variants import expand_cases

DEFAULT_OUTPUT = Path("evals/judge_calibration.yaml")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real agent over the datasets and write a blind labeling file: unique "
            "responses, deterministic synthetic negatives and contrast controls"
        )
    )
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--synthetic-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _paths(output: Path) -> tuple[Path, Path]:
    return output.with_suffix(".manifest.json"), output.with_suffix(".observations.jsonl")


def _refuse_existing_labels(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"{output} already exists; human labels are never overwritten")


def _append_observation(path: Path, observation: CaseObservation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(observation_record(observation), ensure_ascii=False) + "\n")


def _cached_observations(path: Path, *, resume: bool) -> dict[str, CaseObservation]:
    if not path.exists():
        return {}
    if not resume:
        raise FileExistsError(
            f"{path} already exists; use --resume only for an interrupted collection"
        )
    records = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {str(record["case_id"]): observation_from_record(record) for record in records}


async def collect(args: argparse.Namespace) -> Path:
    output: Path = args.output
    manifest_path, observations_path = _paths(output)
    _refuse_existing_labels(output)
    cases = [case for name in args.datasets for case in expand_cases(load_dataset(name))]
    cached = _cached_observations(observations_path, resume=args.resume)
    pending = [case for case in cases if case.id not in cached]
    if pending:
        settings = get_settings()
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required to collect real judge samples")
        llm = build_agent_llm(
            api_key=key,
            model=settings.openai_agent_model,
            check_model=settings.openai_check_model,
        )
        try:
            for index, case in enumerate(pending, 1):
                observed = await run_case(case, llm=llm)
                cached[case.id] = observed
                _append_observation(observations_path, observed)
                print(f"[{index}/{len(pending)}] {case.id}: observada", flush=True)
        finally:
            await llm.aclose()

    real, manifest = unique_real_samples([(case, cached[case.id]) for case in cases])
    synthetic, synthetic_manifest = synthetic_negatives(
        real, ratio=args.synthetic_ratio, seed=args.seed
    )
    controls, control_manifest = vulnerability_controls()
    write_pending_labels(build_label_file(real, synthetic, controls, seed=args.seed), output)
    write_manifest({**manifest, **synthetic_manifest, **control_manifest}, manifest_path)
    authored = sum(bool(item["model_authored"]) for item in manifest.values())
    print(
        f"Muestras: {len(real) + len(synthetic)} · reales únicas: {len(real)} "
        f"(redactadas por el modelo: {authored}) · sintéticas: {len(synthetic)} · "
        f"controles: {len(controls)} · "
        f"Archivo: {output}"
    )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(collect(parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover
    main()
