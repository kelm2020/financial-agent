# Cost regression experiment (prompt caching)


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

## Hallazgo en producción (16/09/2026)

La corrida live del agente real (`make eval-live K=5`, suite canónica, 750 runs) cerró así:

| Campo | Valor |
|---|---|
| `input_tokens` totales | 561.601 |
| `cached_tokens` totales | **0** |
| Costo input no-cached | USD 0,0281 |

**`cached_tokens = 0` significa que el prompt cache de OpenAI no se está aprovechando.** Con las
tarifas de septiembre 2026, el input cacheado vale 10× menos que el input fresco (USD 0,005/M
vs USD 0,05/M); tener cero cacheado es plata que se va en cada conversación.

El experimento simulado (`FakeProvider`) reproduce el patrón con un prefijo estable: la primera
llamada falla el cache, las cuatro siguientes hit 100 %. Con un timestamp arriba del prefijo, las
cinco llamadas fallan.

**Diagnóstico en vivo del agente**: el `OpenAIResponsesLLM` arma `instructions` con partes
fijas (system prompt, política de tools) más partes variables que pueden estar entrando antes del
sufijo cacheable. Sospechosos a inspeccionar:

1. `SYSTEM_PROMPT_CANARY` rotando entre despliegues.
2. Algún timestamp o `conversation_id` que se esté concatenando arriba del bloque estático.
3. El orden de las tools cambiadas en el payload.
4. Bloques por turno (historial, slots) que se anteponen a la política estática.

**Acción**: capturar el `instructions` exacto que se envía a la API en cada llamada, comparar
byte a byte entre dos turnos del mismo `thread_id`, e identificar la primera diferencia. La
corrección es mover lo variable **al final** del `instructions` para que el prefijo cacheable
quede intacto.

**Costo evitable estimado**: con la configuración de la corrida, ~USD 0,025 por conversación
recuperables sólo en input. A escala (miles/día) eso es plata que vuelve al bolsillo sin tocar
calidad.

## Cómo extender al modelo real con OTel

Con la instrumentación OpenTelemetry del runtime, cada llamada LLM emite un span
`gen_ai.inference` con los atributos `gen_ai.usage.input_tokens`, `output_tokens` y
`cached_tokens`. Enviando esos spans a Langfuse (o cualquier backend OTLP), se puede graficar la
evolución del hit rate por tarea a lo largo del tiempo y detectar regresiones del caching sin
necesidad de re-correr el experimento a mano.
