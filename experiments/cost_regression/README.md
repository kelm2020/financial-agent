# Cost regression experiment (prompt caching)

Reproducible demonstration of §12.5's playbook: a volatile timestamp above the cacheable prefix
of the system prompt kills the hit rate, and the cost shows it.

## What it measures

| Variant | System-prompt shape | Expected hit rate | Expected relative cost |
|---|---|---|---|
| `stable` | Static system prompt only | High (≥ 80 % after warm-up) | Baseline |
| `unstable` | `[timestamp=...]` line above the static block | ~ 0 % on the first call after the timestamp changes | Higher (depends on input price vs cached price) |

The numbers are simulated: the experiment does not call the real OpenAI provider. The
`FakeProvider` mirrors the production rules of OpenAI's prompt cache — exact-byte match on the
prefix — and reports `cached_tokens` for each call. The relative cost uses the same per-million
rates the live runner configures.

## How to run

```bash
uv run python -m experiments.cost_regression.run --variant both --runs 5
```

Reports land in `experiments/cost_regression/reports/{stable,unstable}.json` and the console
prints the cost ratio between the two variants.

## How to extend with the real provider

Replace `FakeProvider` with a thin wrapper that intercepts the OpenAI HTTP call and records the
real `usage.input_tokens_details.cached_tokens`. The shape of the report stays identical; only
the data source changes.

## Verdict criteria

- **stable hit_rate < 0.80**: the cacheable prefix is moving even in the "stable" branch;
  look at the order in which the system prompt, tools and business block are composed
  (`app/prompts/`, `app/graph/nodes/respond.py`).
- **unstable hit_rate > 0**: the timestamp is below the cacheable prefix instead of above it.
- **ratio < 1.5x**: the cache price differential is too small to dominate; rerun with the
  provider's actual prices.
