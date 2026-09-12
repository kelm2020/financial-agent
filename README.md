# Agente conversacional de cobranzas

Implementación por fases del challenge técnico de Froneus. El LLM interpreta y redacta;
la identidad, las reglas de negocio, la confirmación y los efectos quedan bajo control
determinista del sistema.

## Estado

F2 implementada sobre F0/F1: motor de políticas determinista, corpus canónico, ingesta
versionada, retrieval híbrido `tsvector` + pgvector con RRF, citas y abstención graduada por
riesgo. Las invariantes INV-1 a INV-21 siguen expresadas como contratos black-box con
`ScriptedLLM`; las que dependen del grafo, RLS o voz quedan como `xfail(strict=True)` hasta F3,
F5 y F7. No hay ningún nodo del grafo todavía.

### Motor de políticas

`app/policy/rules.yaml` es la única fuente de las reglas: límites, medios de pago y reglas de
escalamiento, en orden y con su motivo. El motor rechaza al cargar un archivo inconsistente
(segmentos con huecos, recargos que no cubren todas las cuotas, una señal sin regla). La
decisión siempre recibe un reloj explícito (`as_of` con zona horaria). Cada regla financiera
tiene un caso que viola sólo esa regla, y un mutation test confirma que borrar cualquiera de las
21 reglas probadas pone la suite en rojo. `test_kb_matches_rules` compara contra `rules.yaml`
cada número que citan los cuatro documentos de la KB, y tiene su propio meta-test que prueba que
detecta cambios.

### Retrieval: resultados medidos

Embeddings `text-embedding-3-large` y reranker Cohere `rerank-v3.5`, servidos desde cachés
commiteados: se reproduce sin red ni credenciales. Umbrales, gate de evidencia y activación del
reranker se decidieron **sólo** en el split dev. El split test son las 9 queries del Anexo E.1 y
se evaluó recién después.

| Split | Reranker | Gate | recall@3 | MRR | Abstención |
|---|---|---|---:|---:|---:|
| test | no | sin gate | 0,83 | 0,60 | 0/3 |
| test | rerank-v3.5 | sin gate | 1,00 | 0,81 | 0/3 |
| test | no | denso (0,505) | 0,50 | 0,31 | 3/3 |
| **test** | **rerank-v3.5** | **denso (0,505)** | **0,50** | **0,42** | **3/3** |
| dev | rerank-v3.5 | denso (0,505) | 0,75 | 0,70 | 9/10 |

Valores de Postgres 17 + pgvector; el store en memoria da los mismos resultados (paridad
verificada en integración). **El reranker mejora el ranking** (recall@3 = 1,00 en test), pero
**el criterio de F2 no se cumple**: el gate que abstiene 3/3 también abstiene preguntas
respondibles. Ni la similitud densa ni el score del cross-encoder separan bien las dos clases, y
en dev el gate de reranker quedó peor que el denso. Una versión anterior reportaba 1,00 / 3/3
con sinónimos derivados de las queries de test; esa cifra se retiró. Detalle en
[`evals/reports/retrieval.md`](evals/reports/retrieval.md).

El corpus contiene las **35 secciones autoritativas** del blueprint. El objetivo de 250–350
chunks requiere políticas, casos e históricos aprobados que la especificación no provee; no
se duplicó ni inventó contenido para alcanzar artificialmente ese número.

El `xfail` mantiene útil la suite y el gate de coverage durante esta fase. `make test` es el
único comando de pruebas para todas las fases y reporta explícitamente los contratos
pendientes. Un XPASS es estricto: implementar una invariante sin actualizar su estado también
rompe CI.

## Matriz de invariantes

| ID | Estado en F1 | Tests |
|---|---|---|
| INV-1 | Verde (F0) | test_customer_id_is_not_a_tool_parameter |
| INV-2 | Verde (F0) | test_write_tool_not_exposed_to_model |
| INV-3 | Rojo hasta F3 | test_no_agreement_without_valid_confirmation |
| INV-4 | Rojo hasta F3 | test_executed_draft_is_the_confirmed_draft |
| INV-5 | Rojo hasta F3 | test_expired_draft_is_refreshed_not_executed |
| INV-6 | Rojo hasta F3 | test_llm_cannot_produce_a_yes_verdict |
| INV-7 | Rojo hasta F3 | test_negative_lexicon_beats_affirmative |
| INV-8 | Rojo hasta F3 | test_two_concurrent_confirmations_create_one_agreement |
| INV-9 | Rojo hasta F3 | test_write_unknown_outcome_never_claims_success |
| INV-10 | Rojo hasta F3 | test_no_hallucinated_numbers; test_injected_policy_chunk_cannot_add_phone |
| INV-11 | Rojo hasta F3 | test_cross_customer_idor; test_customer_id_cannot_be_changed_by_language |
| INV-12 | Rojo hasta F3 | test_zero_debt_customer_is_not_escalated; test_not_found_is_not_no_debt |
| INV-13 | Rojo hasta F3 | test_stale_options_are_refreshed |
| INV-14 | Rojo hasta F3 | test_logs_contain_no_pii |
| INV-15 | Rojo hasta F7 | test_t0_has_no_business_tools |
| INV-16 | Rojo hasta F7 | test_t0_discloses_nothing |
| INV-17 | Rojo hasta F7 | test_voice_yes_requires_dtmf |
| INV-18 | Rojo hasta F3 | test_checkpoint_requires_ownership_check; test_foreign_conversation_id_returns_404 |
| INV-19 | Rojo hasta F5 | test_cache_keys_are_customer_scoped |
| INV-20 | Rojo hasta F5 | test_rls_blocks_foreign_customer_at_engine; test_rls_holds_when_application_checks_are_bypassed; test_set_local_does_not_leak_across_pooled_connections |
| INV-21 | Verde (F0) | test_backend_rejects_foreign_sub |

## Requisitos

- Python 3.12 (gestionado automáticamente por `uv`).
- `uv`.
- Docker con Compose para ejecutar Postgres/pgvector, Redis, Langfuse y el mock integrado.

## Uso local (F2)

```bash
cp .env.example .env
make setup
make up
make ingest      # migra e indexa en pgvector con los embeddings cacheados
make test        # unitarios, sin red ni Postgres
make test-rag    # incluye la integración con pgvector, en una base de test propia
make eval-rag    # split test, en memoria y contra el índice de Postgres
```

`data/query_cache.json` y `data/rerank_cache.json` están commiteados. Tests, calibración y
evaluación leen embeddings y scores reales sin red, y **fallan si falta uno**, en vez de cambiar
de espacio en silencio. Sólo dos comandos llaman a proveedores, y hay que correrlos cuando cambia
la KB o un dataset:

- `make embeddings-cache` (requiere `OPENAI_API_KEY`).
- `make rerank-cache` (requiere `COHERE_API_KEY`; reintenta ante 429 respetando `Retry-After`).

`make calibrate-rag` y `make calibrate-rerank` imprimen los umbrales a partir de dev;
`make eval-rag-rerank` evalúa test con reranker. `make coverage` y CI corren la suite de
Postgres: el store de pgvector no tiene exclusión de cobertura.

Servicios locales:

- Mock API y OpenAPI: `http://localhost:8001/docs`
- Langfuse: `http://localhost:3000`
- Postgres/pgvector: `localhost:5432`, base `collections`
- Redis: `localhost:6379`

Para ejecutar el mock sin Docker:

```bash
make mock
```

El endpoint local `POST /auth/token` emite tokens HS256 de cinco minutos únicamente para
demostrar el alcance por cliente. No reemplaza un IdP real.
