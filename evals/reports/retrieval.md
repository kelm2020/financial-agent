# Retrieval F2 — medición con embeddings y reranker reales

Fecha de corrida: 2026-09-12. Embeddings `text-embedding-3-large` (1536 dimensiones) y reranker
Cohere `rerank-v3.5`, servidos desde `data/query_cache.json` y `data/rerank_cache.json`: la
corrida se reproduce sin red ni credenciales. Corpus: las 35 secciones del Anexo E.

## Protocolo

- **test** (`evals/retrieval.yaml`): las 9 queries del Anexo E.1, sin cambios. Es held-out:
  nada se ajusta mirándolo.
- **dev** (`evals/retrieval_dev.yaml`): 32 positivas sobre las 35 secciones, con otras
  formulaciones, y 10 negativas plausibles que la KB no responde.
- **Qué se decide en dev:** los dos umbrales, qué gate de evidencia usar y si se enciende el
  reranker. Test se evaluó después de fijar esas decisiones.
- **Normalización léxica:** plegado de tildes más Snowball Spanish, calculado en Python. Postgres
  guarda esos mismos lexemas con la configuración `simple`, así que los dos stores recuperan el
  mismo conjunto de candidatos léxicos. Hay un test que compara, chunk por chunk, los lexemas
  guardados con los calculados.

## Calibración y decisiones (sólo dev)

| Gate de evidencia | Umbral | Positivas con evidencia | Negativas abstenidas |
|---|---:|---:|---:|
| Similitud densa (vecino más cercano) | 0,505 | 24/32 | 9/10 |
| Score del cross-encoder | 0,209 | 22/32 | 9/10 |

| Configuración (dev, memoria) | recall@3 | MRR | Abstención |
|---|---:|---:|---:|
| Sin reranker, gate denso | 0,72 | 0,67 | 9/10 |
| **Reranker, gate denso** | **0,75** | **0,70** | **9/10** |
| Reranker, gate de reranker | 0,69 | 0,66 | 9/10 |
| Reranker, sin gate | 1,00 | 0,94 | 0/10 |

**Decisión:** el reranker se usa para ordenar y la similitud densa para abstenerse
(`RAG_EVIDENCE_GATE=dense`).

- **Ordenando, el reranker aporta:** con él, las 32 positivas de dev tienen la sección correcta
  en el top-3.
- **Como gate, separa peor que la similitud densa:** asigna scores bajos a respuestas correctas
  formuladas en rioplatense (D-21: 0,039; D-03: 0,056).

## Resultados en test

| Store | Reranker | Gate | recall@3 | MRR | Abstención |
|---|---|---|---:|---:|---:|
| memoria | no | denso 0,505 | 0,50 | 0,33 | 3/3 |
| Postgres 17 + pgvector | no | denso 0,505 | 0,50 | 0,31 | 3/3 |
| memoria | no | sin gate | 0,83 | 0,58 | 0/3 |
| Postgres 17 + pgvector | no | sin gate | 0,83 | 0,60 | 0/3 |
| memoria | rerank-v3.5 | denso 0,505 | 0,50 | 0,42 | 3/3 |
| **Postgres 17 + pgvector** | **rerank-v3.5** | **denso 0,505** | **0,50** | **0,42** | **3/3** |
| memoria | rerank-v3.5 | sin gate | 1,00 | 0,81 | 0/3 |
| Postgres 17 + pgvector | rerank-v3.5 | sin gate | 1,00 | 0,81 | 0/3 |

- **Ranking:** con reranker, recall@3 = 1,00 en test. El orden cumple la parte de recall del
  criterio de F2.
- **Abstención:** el gate abstiene 3/3 negativas, pero también R-02 (evidencia 0,434), R-03
  (0,413) y R-06 (0,268). Por eso recall@3 queda en 0,50. R-06 es de escalamiento, así que su
  abstención devuelve `derivar`, que coincide con ESC-001.

**Criterio de aceptación de F2 (recall@3 = 1,00 con abstención 3/3 al mismo tiempo): no se
cumple.** Con 42 ejemplos de dev, ningún umbral sobre la similitud densa ni sobre el
cross-encoder separa las preguntas que la KB responde de las que no. El siguiente paso natural
es la verificación de respaldo en el generador (F3, §7.4): decidir la abstención con la
respuesta en la mano, no sólo con la similitud del retrieval.
