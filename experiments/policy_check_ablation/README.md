# Policy-check ablation experiment

Measures the cost/quality trade-off of ``policy_answer_check`` (ADR-011). The check is the
semantic gate that decides whether a model-written policy answer gets shown to the customer or
falls back to verbatim quotes of the retrieved sections. It runs on a separate model
(``OPENAI_CHECK_MODEL``) and is the dominant cost of a policy turn.

## Method

The script runs the same canonical suite twice:

1. **with_check**: the production graph path. Each policy turn invokes ``policy_answer_check``
   (LLM call #2 in the turn pipeline) on top of the answer generator.
2. **without_check**: the answer goes directly to the verbatim fallback (no second LLM call).

For each variant we report:

- p50 / p95 / p99 latency per turn.
- Total input / output / cached tokens per conversation.
- Total cost per conversation.
- The safety gates that justify the check: ``policy_compliance``,
  ``grounded_answer_rate`` and ``hallucinated_numbers``.

## How to run

```bash
# Dry-run (no API key required): prints the plan.
uv run python -m experiments.policy_check_ablation.run --dry-run

# Live run: requires OPENAI_API_KEY and consumes provider credits.
uv run python -m experiments.policy_check_ablation.run --dataset evals/cases --k 5
```

Reports land in ``experiments/policy_check_ablation/reports/policy_check_ablation.json``.

## Verdict criteria

The check earns its keep if:

- ``policy_compliance`` drops by **more than 0.02** without it, **or**
- ``hallucinated_numbers`` increases by **more than 0.01**, **or**
- ``grounded_answer_rate`` drops by **more than 0.05**.

Otherwise the recommendation is to lower the check model (or skip it on low-risk topics per
§10.1.6) and reclaim the latency budget.
