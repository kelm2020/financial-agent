"""Reproducible prompt-caching regression experiment (§12.5).

Measures how stable the cacheable prefix of the system prompt is across calls. The real OpenAI
prompt cache keys on the exact bytes of the prefix; a variable timestamp, a changing tool order
or a per-conversation id above the static block kills the hit rate. This experiment makes that
visible without needing a paid account.

Usage::

    uv run python -m experiments.cost_regression.run --variant stable
    uv run python -m experiments.cost_regression.run --variant unstable

Reports land in ``experiments/cost_regression/reports/<variant>.json`` with prefix length, hit
rate and a verdict comparing the two variants.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.llm.openai_responses import OpenAIResponsesLLM  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT_STABLE = (
    "Sos un asistente virtual de cobranzas de Froneus, hablás español rioplatense, "
    "en voseo, sin presionar y sin prometer resultados que no podés garantizar."
)

# §12.5: a volatile timestamp above the stable block moves the cache key and drops hit rate.
SYSTEM_PROMPT_UNSTABLE = f"[timestamp={time.time():.6f}] " + SYSTEM_PROMPT_STABLE


@dataclass
class FakeCall:
    request_id: int
    instructions: str
    model_input: list[dict[str, str]]
    timestamp: float
    cached_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 64


@dataclass
class FakeProvider:
    """Simulates OpenAI's prompt cache: returns cached_tokens for exact-prefix matches only."""

    stable_prefix: bytes
    history: list[FakeCall] = field(default_factory=list)
    output_cost_usd_per_million: float = 0.40
    input_cost_usd_per_million: float = 0.05
    cached_cost_usd_per_million: float = 0.005

    async def __call__(self, json_body: dict[str, object]) -> dict[str, object]:
        instructions = json_body.get("instructions") or ""
        model_input = json_body.get("input") or []
        request_id = len(self.history) + 1
        cached_bytes = 0
        for prev in reversed(self.history):
            if prev.instructions.startswith(self.stable_prefix):
                cached_bytes = len(self.stable_prefix)
                break
        current_prefix = instructions[: len(self.stable_prefix)]
        if current_prefix == self.stable_prefix and cached_bytes > 0:
            cached = len(self.stable_prefix)
        else:
            cached = 0
        input_tokens = max(len(instructions) // 4, 64)
        call = FakeCall(
            request_id=request_id,
            instructions=instructions,
            model_input=model_input if isinstance(model_input, list) else [],
            timestamp=time.time(),
            cached_tokens=cached,
            input_tokens=input_tokens,
            output_tokens=64,
        )
        self.history.append(call)
        return {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"label":"ok"}'}],
                }
            ],
            "usage": {
                "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens,
                "input_tokens_details": {"cached_tokens": call.cached_tokens},
            },
        }

    def total_cost(self) -> float:
        cost = 0.0
        for call in self.history:
            cost += (call.input_tokens - call.cached_tokens) * self.input_cost_usd_per_million / 1e6
            cost += call.cached_tokens * self.cached_cost_usd_per_million / 1e6
            cost += call.output_tokens * self.output_cost_usd_per_million / 1e6
        return cost


async def run_variant(variant: str, *, runs: int = 5) -> dict[str, object]:
    """Send ``runs`` identical requests with the chosen prefix and record the cache hits."""
    prompt = SYSTEM_PROMPT_STABLE if variant == "stable" else SYSTEM_PROMPT_UNSTABLE
    fake = FakeProvider(stable_prefix=SYSTEM_PROMPT_STABLE)
    client = OpenAIResponsesLLM(
        api_key="sk-experiment",
        model="gpt-5-nano",
        max_output_tokens=200,
        client=_AsyncClient(fake),  # type: ignore[arg-type]
    )
    try:
        for index in range(runs):
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"turn {index}"},
            ]
            from pydantic import BaseModel

            class _Schema(BaseModel):
                label: str

            await client.complete(
                task="guard_classifier", messages=messages, response_model=_Schema
            )
    finally:
        await client.aclose()

    cached = [call.cached_tokens for call in fake.history]
    total_input = [call.input_tokens for call in fake.history]
    total_output = [call.output_tokens for call in fake.history]
    report = {
        "variant": variant,
        "runs": runs,
        "input_tokens_per_call": total_input,
        "output_tokens_per_call": total_output,
        "cached_tokens_per_call": cached,
        "hit_rate": _hit_rate(cached),
        "total_cost_usd": fake.total_cost(),
        "system_prompt_first_120_chars": prompt[:120],
    }
    out_path = REPORT_DIR / f"{variant}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _hit_rate(cached: list[int]) -> float:
    """Fraction of calls whose system-prompt prefix hit the cache."""
    if not cached:
        return 0.0
    return sum(1 for value in cached if value > 0) / len(cached)


def _print_summary(stable: dict[str, object], unstable: dict[str, object]) -> None:
    print("=== Prompt-caching regression experiment ===")
    print(f"stable   cost=${stable['total_cost_usd']:.6f} hit_rate={stable['hit_rate']:.2%}")
    print(f"unstable cost=${unstable['total_cost_usd']:.6f} hit_rate={unstable['hit_rate']:.2%}")
    if stable["total_cost_usd"] and unstable["total_cost_usd"]:
        ratio = float(unstable["total_cost_usd"]) / float(stable["total_cost_usd"])
        print(f"ratio (unstable / stable) = {ratio:.2f}x")


class _AsyncClient:
    """Tiny httpx.AsyncClient-shaped wrapper around the FakeProvider."""

    def __init__(self, fake: FakeProvider) -> None:
        self._fake = fake

    async def post(self, path: str, json: dict[str, object]) -> _FakeResponse:
        payload = await self._fake(json)
        return _FakeResponse(payload)


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


async def _main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("stable", "unstable", "both"),
        default="both",
        help="Which variant to run. ``both`` runs both and prints the cost ratio.",
    )
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    if args.variant in ("stable", "both"):
        stable = await run_variant("stable", runs=args.runs)
    else:
        stable = json.loads((REPORT_DIR / "stable.json").read_text())
    if args.variant in ("unstable", "both"):
        unstable = await run_variant("unstable", runs=args.runs)
    else:
        unstable = json.loads((REPORT_DIR / "unstable.json").read_text())
    if args.variant == "both":
        _print_summary(stable, unstable)


if __name__ == "__main__":
    os.environ.setdefault("OPENAI_API_KEY", "sk-experiment")
    asyncio.run(_main_async())
