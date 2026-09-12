# Agente conversacional de cobranzas

Implementación por fases del challenge técnico de Froneus. El LLM interpreta y redacta;
la identidad, las reglas de negocio, la confirmación y los efectos quedan bajo control
determinista del sistema.

## Estado

F1 implementada: las invariantes INV-1 a INV-21 están expresadas como contratos black-box
ejecutables y usan `ScriptedLLM`, sin red ni API key. INV-1, INV-2 e INV-21 ya están verdes
por la infraestructura de F0; las que dependen del grafo, RLS o voz quedan como
`xfail(strict=True)` hasta F3, F5 y F7 respectivamente. No hay ningún nodo del grafo todavía.

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

## Uso local (F1)

```bash
cp .env.example .env
make setup
make up
make test
```

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
