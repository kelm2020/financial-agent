# Agente conversacional de cobranzas

Implementación por fases del challenge técnico de Froneus. El LLM interpreta y redacta;
la identidad, las reglas de negocio, la confirmación y los efectos quedan bajo control
determinista del sistema.

## Estado

F0 implementada: infraestructura local, migración inicial, contratos tipados,
`CustomerScope`, gateway resiliente y mock de las cinco APIs de negocio con token acotado,
inyección de fallas e idempotencia completa. El grafo se incorpora recién en F3, después de
escribir las invariantes en rojo durante F1.

## Requisitos

- Python 3.12 (gestionado automáticamente por `uv`).
- `uv`.
- Docker con Compose para ejecutar Postgres/pgvector, Redis, Langfuse y el mock integrado.

## Uso de F0

```bash
cp .env.example .env
make setup
make up
make test-f0
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
