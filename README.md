# Agente conversacional de cobranzas

## Challenge técnico de Froneus · Senior GenAI Engineer

Agente que conversa con un cliente en mora, consulta su deuda, ofrece alternativas dentro de
política, registra un acuerdo con confirmación explícita y deriva a un operador humano cuando corresponde.

Partí del identificador de cliente (`CUST-00125`) que propone el challenge y lo vinculé a la sesión: ni el
modelo ni un mensaje del usuario pueden elegir sobre qué cuenta opera.

### Enfoque de ingeniería: diseño orientado a producción

El proyecto fue concebido desde el inicio con criterios de producción, estructurando el desarrollo en fases iterativas para priorizar la mitigación de riesgos críticos de negocio, la integridad transaccional y la observabilidad en función del tiempo disponible.

Cuenta con una interfaz web en Gradio para facilitar las pruebas manuales y observar las decisiones del agente en tiempo real, la cual cuenta con botones pare responder más rapido o iniciar un nueva conversación.

![Interfaz de prueba](docs/assets/image.png)

El proyecto consta de 8 fases:

## Cómo dividí el trabajo: de F0 a F8

| Fase | Qué abordé | Cómo daré por cerrada la fase |
|---|---|---|
| **F0 · Infraestructura, contratos y mocks** | Compose, migraciones, contratos tipados, `CustomerScope` y un backend con cuatro clientes, cinco integraciones, fallas inyectables e idempotencia. | Reemplazaré mocks e identidad local antes de operar con cuentas reales, validando cada adaptador con los mismos contratos. |
| **F1 · Invariantes** | Límites de identidad, confirmación, escritura, salida y concurrencia expresados en pruebas. Conservo siete `xfail` explícitos. | Convertiré los pendientes en pruebas que pasen en F5 y F7. No cuento un `xfail` como una protección implementada. |
| **F2 · Conocimiento y políticas** | Reglas deterministas, cuatro documentos, ingesta, búsqueda híbrida, citas y evaluación de retrieval. | Ampliaré el corpus sólo con material aprobado y mediré recuperación, respondibilidad y respuesta final en conjunto. |
| **F3 · Agente de chat** | Grafo async, acuerdos en dos fases, medio de pago elegible, contexto, streaming validado, locks, auditoría y derivaciones. | Cerrado para este corte: la suite aplicable pasa y los gates live se reproducen con configuración congelada. |
| **F4 · Evaluación** | Suites canónica, held-out y ciega, pruebas de políticas, guardrails, simulador, judge calibrado y `pass^k`, con el runner concurrente y métricas p95 por nodo y por tarea. UI para facilitar testeo(más comodo) | Cerrada para este corte: cada reporte ya publica su `prompt_fingerprint` como hash de configuración. Pendiente real, sin bloquear el corte: ampliar las etiquetas humanas del camino generativo y resolver C-51 (sin resolver hoy, ver [Límites que conservo visibles](#límites-que-conservo-visibles)). |
| **F5 · Seguridad y aislamiento** | Controles de aplicación, JWT local, protección de entradas y salidas y auditoría con HMAC. | **Diferida.** Implementaré RLS con rol no dueño, `FORCE ROW LEVEL SECURITY` y `SET LOCAL`; probaré bypass de controles de aplicación. Completaré IdP/OBO, aislamiento de caché, PII, cifrado y retención. Exigiré esta fase antes de usar datos reales. |
| **F6 · Producción medida** | Persistencia, serialización, métricas de evaluación, latencia por ruta y por etapa, instrumentación OpenTelemetry en el runtime y exporter OTLP/HTTP hacia Langfuse. | **Diferida.** Mediré carga, fallos y límites distribuidos con OTel conectado; integraré el experimento de regresión de caching en CI nightly. No asumo escala por tener un servidor async. |
| **F7 · Voice AI / Realtime** | Documentada la evolución; tres invariantes pendientes. Admito sólo el canal chat. | **Diferida.** Primero identidad por niveles, confirmación robusta y continuidad entre llamadas, testeables sin audio; después STT, TTS, interrupciones y transferencia. |
| **F8 · Entrega** | Este README, con ejecución, arquitectura, decisiones, resultados y pendientes. |  |


## Cómo levantar el entorno

Hay **dos formas** de levantar el entorno. La diferencia no es "uno anda y el otro no": ambos
responden preguntas de RAG si tienen `OPENAI_API_KEY`. La diferencia es **persistencia e
infraestructura**.

| | `make run` + `make mock` + `make ui` | `make up` (Compose) + `make ui`|
|---|---|---|
| Persistencia | En memoria (checkpoints, conversaciones, índice RAG) | Postgres + pgvector (durable) |
| Mock backend | Proceso local en `:8001` | Contenedor `:8001` |
| Levantar | 5-10 s, sin Docker | 2-3 min (incluye migración + ingest) |
| Reset | `Ctrl+C` y volver a empezar | `docker compose down -v` |
| RAG funciona | ✅ si hay `OPENAI_API_KEY` | ✅ si hay `OPENAI_API_KEY` |
| Necesario para | Demo conversacional, hacer cambios rápidos | Eval con concurrencia, OTel/Langfuse, persistencia real |
| Concurrencia entre conversaciones | Sí (en memoria) | Sí (Postgres + advisory lock) |
| OpenTelemetry → Langfuse | Opcional vía `OTEL_EXPORTER_OTLP_ENDPOINT` | Recomendado, con `langfuse` como servicio |


### Stack completo con Docker Compose (`make up`)

**Receta de cero** (la uso yo misma cada vez que dudo del estado del stack):

```bash
docker compose down            # baja todo; evita arrastrar un volumen o proceso viejo
make up                        # Postgres/pgvector, Redis, Langfuse, mock y agente
make ingest                    # migra el schema e indexa la KB en pgvector — no es opcional
curl localhost:8000/health     # {"status":"ok","rag_enabled":true,"kb_chunks":35} esperado
make ui                        #Levanta la UI en http://localhost:7860/
```

`rag_enabled` en `false`, o `kb_chunks` en `0` o ausente, dice exactamente qué falta — ver el
detalle de cada caso más abajo. Con eso confirmado:

```bash
make coverage                                             # suite completa con Postgres
make eval-live K=5 JUDGE_MODEL=gpt-4.1-mini-2025-04-14     # evaluación con el modelo real
```

Si ya usás los puertos 8000 y 8001 con la demo local, detené esos procesos antes. **No interpreto
`APP_ENV=production` como certificación productiva:** con esa configuración activo la persistencia,
pero sigo usando autenticación y backend de demostración. **Lo que sí cambia respecto a `make run`:
los checkpoints, conversaciones y el índice de RAG son durables, hay Redis para rate limit
distribuido, y Langfuse queda disponible en `http://localhost:3000` (login
`admin@example.local` / `change-me-now`)**.

### IMPORTANTE

El mismo gotcha de arriba aplica acá: si editás `.env` con el contenedor `agent` ya arriba,
volvé a correr `make up` para que Compose lo recree con las variables nuevas (si sólo cambió
`.env` y nada del compose file, a veces no detecta la diferencia; `docker compose up -d --force-recreate agent`
lo garantiza). `curl localhost:8000/health` sigue siendo la forma rápida de confirmar
`rag_enabled` antes de asumir que el RAG está roto.

**Segundo gotcha específico de este modo, más fácil de pisar todavía: `rag_enabled: true` no
significa que el índice tenga filas.** Si recreaste el volumen de Postgres (`docker compose down
-v`, o un `make up` de cero) y todavía no corriste `make ingest`, la tabla `kb_chunks` está vacía:
el objeto retriever se construye igual (por eso `rag_enabled` da `true`), pero cada búsqueda
devuelve cero resultados y el agente responde la misma abstención genérica de siempre — no hay
forma de distinguirlo de una respuesta legítima sin mirar el dato. Por eso `/health` también
expone `kb_chunks` (cuenta real de `app.rag.store.HybridStore.metadata()`, no una suposición):

```bash
curl localhost:8000/health
# {"status":"ok","rag_enabled":true,"kb_chunks":35}   ← esperado después de `make ingest`
# {"status":"ok","rag_enabled":true,"kb_chunks":0}    ← retriever armado, índice vacío: corré `make ingest`
```

Encontré esta combinación de dos causas (proceso viejo + índice sin ingestar) reproduciendo el
reporte real de "el RAG no responde" con `make up`: cada una por separado ya alcanza para el mismo
síntoma, así que si `kb_chunks` da 0 el fix es `make ingest`, y si `rag_enabled` da `false` el fix
es reiniciar el proceso — nunca tocar el código de retrieval.


### Entorno con (`make run` + `make mock` + `make ui`)

Si no tenés `.env`, copiá `.env.example`. Para conversar con el modelo necesitás
`OPENAI_API_KEY` (más `OPENAI_AGENT_MODEL=gpt-5-nano` y opcionalmente `COHERE_API_KEY` para el
reranker).

```bash
make mock                      # terminal 1: backend simulado en :8001
make run                       # terminal 2: agente en :8000
make ui                        # UI web en http://127.0.0.1:7860
make chat CUSTOMER=CUST-00125  # Chat por terminal

```

> **Si agregaste `OPENAI_API_KEY` o `COHERE_API_KEY` a `.env` con el servidor ya corriendo,
> reiniciá `make run` (y `make up` si es el caso).** `Settings` se cachea una vez por proceso
> (`config/settings.py:get_settings`, `@lru_cache`) y el retriever se construye una única vez al
> arrancar (`app/main.py`, dentro del `lifespan`); `uvicorn --reload` sólo observa cambios en
> archivos `.py`, no en `.env`, así que un proceso que arrancó sin la clave se queda **para
> siempre** sin RAG aunque después completes el `.env`. La señal es indistinguible de una
> abstención real: toda pregunta de política responde lo mismo ("No tengo información
> confirmada... Te derivo con un asesor"), sin ningún log ni error visible — encontré este bug
> exacto reportado como "el RAG no responde" y la causa era un proceso viejo. Para confirmarlo sin
> adivinar, `curl localhost:8000/health` expone `"rag_enabled": true|false`; si da `false` con la
> clave puesta en `.env`, el fix es reiniciar el proceso, no tocar código.

Con `make run` las conversaciones y el índice RAG viven en memoria. Reiniciá el mock para repetir
un cierre con el mismo cliente (guarda acuerdos en memoria). Las fechas de los fixtures se anclan
al día de ejecución para evitar ofertas vencidas por la fecha del fixture.

**El RAG funciona acá si hay `OPENAI_API_KEY`**: el agente ingiere la KB en memoria al arrancar
y las respuestas de política se generan con el mismo `grounded_response` + `policy_answer_check`
que en producción. Sin clave, los caminos deterministas funcionan (consulta de saldo, derivación,
confirmación) pero el clasificador y el RAG quedan deshabilitados: una pregunta de política responde
"no encontré evidencia".

## Testeo

### 4. Evaluación con el modelo real (opcional, consume créditos)

```bash
make eval-live K=5 JUDGE_MODEL=gpt-4.1-mini-2025-04-14
make eval-live K=5 DATASET=heldout JUDGE_MODEL=gpt-4.1-mini-2025-04-14
make eval-live K=5 DATASET=blind JUDGE_MODEL=gpt-4.1-mini-2025-04-14
make eval-sim
make eval-policy LIVE=1 DATASET=evals/policy_blind.yaml
```

Para el juez y el simulador, configurá `OPENAI_JUDGE_MODEL` y `OPENAI_SIMULATOR_MODEL` con un
modelo distinto al del agente (auto-preferencia). Resultados: ver la sección
"Latencia y costo medidos" más abajo — `pass^k` 149/150, p95 2,7 s.

### Sin claves ni servicios externos (la suite entera corre gratis)

Necesitás `uv` y el repositorio clonado:

```bash
make setup
make test                       # 875 passed, 19 skipped, 7 xfailed
make eval                       # canónica 150/150, gates PASS
make eval-heldout               # held-out 30/30, gates PASS
make eval-blind                 # ciega 32/32, perfil de seguridad PASS
make eval-guardrails            # dev y test, 0 escapes
make calibrate-judge SPLIT=test # acuerdo judge-humanos: κ 0,844
```

En estas pruebas uso dobles deterministas y transportes locales; no hago llamadas pagas. Con
`make calibrate-judge` recalculo el acuerdo sobre etiquetas y puntuaciones guardadas, no ejecuto
un judge nuevo. **Las suites de este modo inyectan evidencia controlada al retriever** —no son
un test del RAG de producción sino de la lógica del agente.

## Evaluación y resultados

Evalúo en capas: contratos e invariantes; trayectorias del agente con `ScriptedLLM`; retrieval con
sus propios datasets; conversaciones con modelo real; y un judge contrastado con etiquetas humanas.
No uso una conversación convincente como sustituto de estas mediciones.

Todos los resultados corresponden al mismo corte de código (2026-09-16).

### 1. Offline, sin modelo (contratos, invariantes y seguridad determinista)

| Qué ejecuté | Alcance / Split | Muestras / Tests | Qué obtuve | Veredicto / Gates |
|---|---|---:|---|---|
| **`make test`** | Pruebas unitarias y contratos locales | 901 tests | **875 passed**, 19 skipped, 7 xfailed (40,14 s) | **PASS** (7 `xfail` en F5/F7) |
| **`make eval`** | Suite canónica (`canonical`, k=1) | 150 casos | **150/150** (1.000) · p95 turno: 7,1 ms | **PASS** (0 alucinaciones) |
| **`make eval-heldout`** | Suite held-out (`heldout`, k=1) | 30 casos | **30/30** (1.000) · p95 turno: 8,8 ms | **PASS** (0 alucinaciones) |
| **`make eval-blind`** | Suite ciega (`blind`, k=1) | 32 casos | **32/32** (1.000) · p95 turno: 9,6 ms | **PASS** (seguridad nivel A) |
| **`make eval-guardrails`** | Guardrails (split dev + test) | 193 inputs, 57 outputs | **18/18** ataques detectados · **0/160** falsos bloqueos · **0/33** escapes de salida | **PASS** (0 escapes) |
| **`make calibrate-judge`** | Calibración LLM-Judge vs. humanos | 51 muestras (`split=test`) | Acuerdo global **0,922** · TPR 0,846 · TNR 1,000 · **κ = 0,844** | **PASS** (alta concordancia) |

### 2. Live con `gpt-5-nano`, `pass^5` (evaluación estocástica de punta a punta)

`make eval-live K=5` sobre las tres suites, judge `gpt-4.1-mini-2025-04-14`.

| Métrica | Canónica | Held-out | Ciega |
|---|---:|---:|---:|
| **`pass^5`** (casos que pasan las 5 repeticiones) | **150/150** | **28/30** | **32/32** |
| ejecuciones que pasan | 750/750 | 146/150 | 160/160 |
| números alucinados | **0/1045** | **0/190** | **0/200** |
| acción automática insegura | **0/200** | **0/30** | **0/40** |
| confirmación salteada | **0/50** | **0/5** | — |
| cumplimiento de política | 645/645 | 115/115 | 120/120 |
| respuestas con respaldo | 55/55 | 10/10 | — |
| escalamiento recall · precisión | 200/200 · 200/200 | 60/60 · 60/60 | 80/80 · 80/80 |
| argumentos de tools válidos | 1895/1895 | 245/245 | 240/240 |
| `tool_selection_f1` | 1.000 | 1.000 | 1.000 |
| judge, todos los criterios | 936/1045 | 180/190 | 175/200 |
| costo | USD 0,093 | USD 0,018 | USD 0,010 |
| **gates** | **PASS** | `case_expectations` | **PASS** |

**Todos los gates de seguridad pasan en las tres suites.** El único que falla es
`case_expectations` en held-out: C-51 ("¿Puede pagar otra persona por mí?") deja de mencionar
"transferencia" en 4 de 5 repeticiones porque el chequeo semántico poda ese claim. Es una respuesta
incompleta, no incorrecta, y está sin resolver.

Entiendo `pass^5` como casos que aprueban sus cinco ejecuciones, no como una potencia del promedio.

### 3. Síntesis operacional en vivo (SLA y costos)

| Métrica operativa | Resultado consolidado | Referencia |
|---|---|---|
| **`pass^k` global (canónica)** | **150/150** | 750 ejecuciones |
| **p95 por turno** | **2,7 s** | Umbral de diseño: < 4,0 s |
| **Costo por conversación** | **USD 0,0695** | Sin considerar llamadas del judge |
| **Tasa de números alucinados** | **0,0%** (0 / 1.435 total suites) | Invariante crítico |
| **Bypass de confirmación** | **0,0%** (0 / 55 total suites) | Invariante crítico |
| **Acciones automáticas inseguras** | **0,0%** (0 / 270 total suites) | Invariante crítico |

Detalle y desglose por nodo/tarea LLM en [Latencia y costo medidos](#latencia-y-costo-medidos).


## Latencia y costo medidos

La **corrida live** con `gpt-5-nano` + judge `gpt-4.1-mini-2025-04-14` sobre la suite canónica
(`make eval-live K=5 JUDGE_MODEL=gpt-4.1-mini-2025-04-14`) cerró así:

| Métrica | Resultado |
|---|---|
| **p95 por turno** | **2.685 ms** (debajo de 4 s) ✅ |
| `pass^k` (5 repeticiones) | **149/150** (0,993) |
| `unsafe_auto_action` | 0/200 |
| `confirmation_bypass` | 0/50 |
| `hallucinated_numbers` | 0/1.045 |
| `policy_compliance` | 644/645 (0,998) |
| `escalation_recall` / `precision` | 200/200 · 200/201 |
| Tokens input / output por conversación | 561.601 / 103.488 |
| **Costo por conversación** (sin judge) | **USD 0,0695** |

### Latencia por nodo (p95 ms, nodo de la traza del grafo)

| Nodo | p95 |
|---|---:|
| `guard_classifier` | **1.798** |
| `execute_agreement` | 18,4 |
| `build_draft` | 14,3 |
| `escalate` | 11,7 |
| `hydrate` | **9,1** |
| `confirm_gate` | 8,6 |
| `render_and_validate` | 5,0 |
| `resolve_guard` | 1,4 |
| `plan_response` | 1,3 |
| `route_or_confirm` | 0,7 |
| `guard_rules` | 0,1 |

### Latencia por tarea LLM (p95 ms)

| Tarea | p95 |
|---|---:|
| `grounded_response` | 8.334 |
| `policy_answer_check` | 6.480 |
| `route` | 1.814 |
| `guard_classifier` | 1.798 |
| `confirmation` | 1.359 |

**Lectura**: el p95 del turno (2,7 s) está debajo del umbral del challenge porque no todos los turnos
ejercen el camino de política (que es el más caro: `grounded_response` + `policy_answer_check` en
serie). El cuello de botella residual **no es el backend**: el nodo `hydrate` cierra en 9 ms tras
paralelizar las tres lecturas (`get_customer`, `get_debt`, `get_payment_options`) en un único
`asyncio.gather`. Lo son dos llamadas LLM seriales del camino de política, dominadas por la
generación con razonamiento (`grounded_response`).

### Por qué el p95 total es 2,7 s y no 13 s

Las cinco rutas tienen estos p95 observados:

| Ruta | p95 turno |
|---|---:|
| saldo (plantilla) | ~1,8 s |
| opciones (plantilla) | ~2,1 s |
| fuera de dominio | ~1,7 s |
| derivación | ~1,8 s |
| **política (RAG + chequeo)** | **~13 s cuando se ejecuta completo** |

El **p95 agregado** del turno (2,7 s) es menor que el p95 de política porque la política es un
camino minoritario: la mayoría de turnos cierra con `route + plan_response + render_and_validate`,
sin RAG. Una pregunta de política puntual puede llevar 13 s y eso es la métrica correcta cuando
se evalúa ese camino, pero el turno promedio —la que ve el cliente y la que mide el SLA— es 2,7 s.


## Por qué prioricé en este orden

Primero lo que puede producir un error difícil de revertir: consultar otra cuenta, ofrecer
condiciones inválidas o registrar un acuerdo sin consentimiento. Por eso implementé contratos,
políticas deterministas, confirmación e idempotencia antes de optimizar la conversación. Después
incorporé recuperación de conocimiento y evaluación, para medir seguridad y capacidad de resolver
la gestión. Reservé escala y voz para después de estabilizar ese núcleo.

El criterio detrás de dónde corté (F4) no es sólo tiempo: es **qué riesgo se ejercita en cada turno
de esta demo, contra qué riesgo sólo existiría en una forma de despliegue distinta a la que estoy
entregando.** Identidad, confirmación, política y evaluación se ejercitan en cada mensaje que
alguien manda hoy — por eso están adentro de F0-F4 y medidos. Aislamiento en el motor de base de
datos, carga concurrente real y un segundo canal de voz son riesgos reales, pero de un despliegue
con credenciales de producción, tráfico real y telefonía — ninguno de los cuales es el que este
challenge ejercita. Meterles tiempo antes de tener F0-F4 probado y medido habría sido optimizar
una amenaza que todavía no existe en este entorno, a costa de dejar sin medir lo que sí se prueba
en cada turno. El detalle de **por qué cada una y cómo la resolvería** está más abajo, en
["Por qué no prioricé F5, F6 y F7"](#por-qué-no-prioricé-f5-f6-y-f7-y-cómo-las-resolvería).

Para el corte sugerido de cuatro horas habría priorizado un recorrido completo y demostrable:
mock, política, consulta, propuesta, confirmación, RAG acotado y casos de prueba. **Mi desarrollo
supera ese corte; no lo presento como una implementación hecha en cuatro horas.** Elegí profundizar
el mismo problema con persistencia, guardrails y evaluación. También agregué una interfaz de demo,
que la consigna no exige, para facilitar las pruebas y grabar evidencia.


## Por qué no prioricé F5, F6 y F7 (y cómo las resolvería)

**F5 · Seguridad y aislamiento en base (RLS, IdP real, cifrado, retención)**

*Por qué no la prioricé:* el aislamiento que protege este challenge hoy —`CustomerScope` armado
desde la sesión, ownership verificado en cada lectura y escritura, el propio backend
re-validando el `sub` del token de forma independiente (`test_backend_rejects_foreign_sub`, ya
verde, no `xfail`)— cubre la superficie de ataque real de esta demo: un IdP local de un solo
emisor y cuentas ficticias. RLS en el motor es defensa en profundidad contra un escenario
distinto: una consulta de la capa de aplicación con un bug que se salta el `WHERE`. Ese es un
riesgo real cuando hay credenciales de producción y varios tenants compartiendo el mismo
Postgres — no el que este challenge ejercita. Invertir tiempo ahí antes de tener F0-F4 probado y
medido habría sido blindar una amenaza que todavía no existe en este despliegue, a costa de dejar
sin medir lo que sí se ejercita en cada turno.

*Cómo la resolvería:* un rol de Postgres sin `BYPASSRLS`, `FORCE ROW LEVEL SECURITY` en cada
tabla con datos de cliente, `SET LOCAL app.customer_id` al abrir cada conexión del pool y una
policy que la compara contra la fila. Los 4 tests que ya escribí —`test_rls_blocks_foreign_customer_at_engine`,
`test_rls_holds_when_application_checks_are_bypassed`, `test_set_local_does_not_leak_across_pooled_connections`,
`test_cache_keys_are_customer_scoped`— son la definición de "hecho": dejan de tener `@RED_F5` y
tienen que pasar en verde antes de operar con datos reales. Sumaría IdP/OBO real (reemplazando el
emisor local), cifrado de PII en reposo y una política de retención y borrado.

**F6 · Producción medida (carga, fallos distribuidos, routing de modelos)**

**A diferencia de F5 y F7, esta fase no está diferida — está parcial.** Ya construí y uso la mitad
de instrumentación que la fase pide: OTel con spans por turno, nodo, tool y llamada LLM; p95 por
nodo y por tarea; costo por conversación; el experimento de regresión de prompt-caching. Son los
números que este mismo README reporta en [Latencia y costo medidos](#latencia-y-costo-medidos). Lo
que falta específicamente es la prueba de carga con tráfico concurrente real — eso es lo que sigue
sin cerrar, no la fase completa.

*Por qué no cerré también la prueba de carga:* correrla necesita un perfil de tráfico y un SLO que
el challenge no define: inventar una cifra arbitraria de usuarios concurrentes daría un número sin
un objetivo real contra el cual validarlo, y cada corrida quema presupuesto de API real con poca
señal nueva (la varianza del modelo ya la mide `pass^k` con K=5; una prueba de carga estresa sobre
todo mi propio código async, no al LLM). Prioricé tener la medición completa y correcta —la que ya
está construida— antes que ejecutar una prueba que no sabría contra qué comparar.

*Cómo la resolvería:* correría k6 o locust contra el stack de Compose con tráfico simultáneo del
mismo cliente (para ejercitar el advisory lock) y de clientes distintos (para ejercitar el
circuit breaker y el rate limiter), publicaría p50/p95/p99 bajo carga junto a los números
secuenciales que ya están en este README, y conectaría el experimento de cost-regression a un job
nightly de CI para alertar una regresión de caching antes de que llegue a la factura.

**F7 · Voice AI / Realtime**

*Por qué no la prioricé:* elegí escribir primero los tres invariantes que capturan el riesgo
específico de voz —nada de datos de negocio antes de verificar identidad en T0, confirmación
sólo por DTMF porque una voz no tiene la misma garantía que una sesión autenticada por texto—
como especificación ejecutable, antes de construir el canal. No construí el pipeline de audio
(STT/TTS, interrupciones, transferencia en caliente) porque es una superficie de ingeniería
genuinamente distinta —infraestructura de telefonía e integración con proveedores de ASR/TTS—
que habría consumido el presupuesto completo en algo ortogonal a lo que este challenge evalúa: la
negociación, el grounding de política y los guardrails, que sí son el núcleo.

*Cómo la resolvería:* primero identidad por niveles y continuidad entre llamadas, testeables sin
audio (ya lo permiten los 3 `xfail`: `test_t0_has_no_business_tools`, `test_t0_discloses_nothing`,
`test_voice_yes_requires_dtmf`); después integraría un proveedor de STT/TTS, manejo de
interrupciones (barge-in) y transferencia en caliente a un humano sin perder el contexto de la
conversación.

Reconozco las desviaciones deliberadas: agregué UI, mantuve un corpus pequeño y aumenté el
presupuesto a cuatro llamadas al modelo para poder verificar la respuesta, decisión justificada en
[ADR-011](docs/decisions/ADR-011-policy-answers-and-eval-integrity.md).


### Estado del challenge

| Fase | Estado |
|---|---|
| F0 · Infraestructura, contratos y mocks | Hecho |
| F1 · Invariantes | Hecho (31 tests: 24 verdes, 7 `xfail` -> test para cuando implemente las fases 5 y 7) |
| F2 · Políticas y RAG híbrido medido | Hecha con el trade-off de recall/abstención documentado |
| F3 · Agente | Hecho |
| F4 · Evaluación y UI| Hech0 y medido con `pass^5` live |
| F5 · Aislamiento en base (RLS, token OBO) | **Diferida** — 4 tests `xfail` |
| F6 · Producción medida | Parcial: OTel conectado al runtime; falta load test — **0 tests** todavía |
| F7 · Voz | **Diferida** — sólo canal chat, 3 tests `xfail` |

> Para el corrector: la diferencia entre "implementado" y "diferido" es explícita en cada fila.
> Un test `xfail` no cuenta como protección implementada, sólo como una especificación pendiente (son test hechos para fases del proyecto a implementar a futuro pero que realice desde el momento 0 siempre pensando en encarar el proyecto por fases hasta produccion).
> Detalle de qué prueba cada uno, por qué no se priorizó y cómo lo resolvería en
> ["Por qué no prioricé F5, F6 y F7"](#por-qué-no-prioricé-f5-f6-y-f7-y-cómo-las-resolvería).

<details>
<summary><strong>Qué prueba cada <code>xfail</code></strong> (7 tests, todos en <code>tests/test_invariants.py</code>)</summary>

| Fase | Test | Qué especifica |
|---|---|---|
| F5 | `test_rls_blocks_foreign_customer_at_engine` | Una fila de otro cliente no vuelve aunque la query de aplicación no filtre por `customer_id` |
| F5 | `test_rls_holds_when_application_checks_are_bypassed` | El aislamiento se sostiene incluso si el chequeo de la capa de aplicación se saltea |
| F5 | `test_set_local_does_not_leak_across_pooled_connections` | `SET LOCAL app.customer_id` no sobrevive ni se filtra a la siguiente conexión que toma del pool |
| F5 | `test_cache_keys_are_customer_scoped` | Las claves de caché llevan el `customer_id` como parte de la clave, nunca compartidas entre cuentas |
| F7 | `test_t0_has_no_business_tools` | En el primer turno de voz (T0), antes de verificar identidad, no se ejecuta ninguna tool de negocio |
| F7 | `test_t0_discloses_nothing` | T0 no revela deuda ni datos de cuenta en la respuesta, aunque se pregunten explícitamente |
| F7 | `test_voice_yes_requires_dtmf` | Un "sí, confirmo" hablado no alcanza para registrar un acuerdo: hace falta confirmación DTMF |

F6 no tiene ningún test todavía — ni siquiera un `xfail` — porque no llegué a escribir los
invariantes de esa fase (ver la sección de abajo).

</details>


## Arquitectura y decisiones

```mermaid
flowchart TD
    A["Recibo un mensaje por CLI o UI"] --> B["Valido sesión, ownership y entrada en FastAPI"]
    B --> C["Serializo la conversación y ejecuto LangGraph"]
    C --> D["Aplico guardrails y resuelvo intención o confirmación"]
    D --> E["Consulto cliente, deuda y opciones en el mock"]
    D --> F["Recupero políticas de la KB"]
    E --> G["Valido condiciones con mi motor de políticas"]
    G --> H["Preparo un borrador y solicito confirmación"]
    H --> I["Revalido y registro con idempotencia"]
    D --> J["Derivo cuando corresponde"]
    F --> K["Genero con citas y verifico el respaldo"]
    G --> L["Valido la respuesta antes de mostrarla"]
    H --> L
    I --> L
    J --> L
    K --> L
    L --> M["Emito cláusulas por SSE y conservo contexto"]
    I --> N["Audito la acción y su resultado"]
    J --> N
```

| Mi decisión | Cómo la implementé y por qué |
|---|---|
| Controlo el flujo explícitamente | Python, FastAPI async y un grafo LangGraph. Resuelvo rutas previsibles en código y recurro al LLM para clasificación o redacción. Puedo inspeccionar la trayectoria y acotar el trabajo por turno. |
| Separo reglas de explicaciones | [rules.yaml](app/policy/rules.yaml) y un motor determinista validan importes y condiciones; el RAG explica políticas. No delego la autorización de un plan al prompt. |
| Controlo la escritura | Preparo un borrador, muestro sus términos y requiero una afirmación reconocida por código. El LLM sólo distingue rechazo de duda. Antes del POST releo y revalido; uso idempotencia para evitar duplicados. |
| Acoto la identidad | Construyo `CustomerScope` desde la sesión y verifico la propiedad de la conversación. No expongo `customer_id` ni `create_payment_agreement` como decisiones libres del modelo. |
| Valido antes de emitir | Centralizo la salida en `render_and_validate`, con controles de cifras, citas, contactos y contenido. Emito cláusulas validadas por SSE, nunca tokens crudos del proveedor. |
| Conservo estado y trazabilidad | Memoria en local y `AsyncPostgresSaver` en modo persistente, locks por conversación y auditoría. Mantengo ocho turnos y un resumen validado; releo los datos antes de acciones sensibles. |
| Desacoplo el proveedor | `LLMClient`, el adaptador real y `ScriptedLLM`. Configuro agente y chequeo por separado; no afirmo haber implementado un router multiproveedor. |

### Las cinco integraciones del challenge

| Integración | Cuándo la uso |
|---|---|
| `get_customer` | Condiciones del cliente necesarias para gestionar su cuenta. |
| `get_debt` | Saldo, vencimientos y estado; distingo saldo cero de dato inexistente. |
| `get_payment_options` | Alternativas, filtradas por mi política antes de ofrecerlas. |
| `create_payment_agreement` | Registro únicamente desde el flujo controlado, después de confirmar y revalidar. **No está en el catálogo que ve el modelo.** |
| `transfer_to_human` | Derivo por pedido explícito, reclamo, vulnerabilidad, restricciones de negocio o una falla que impida continuar con seguridad. |

Valido entradas y respuestas con contratos tipados. Aplico timeouts, reintentos acotados y circuit
breaker; no sigo usando datos obsoletos si falla su actualización. Si un registro queda incierto,
no afirmo éxito ni fracaso: conservo la identificación de la operación y derivo para reconciliar.
Cubro fallas y respuestas incompletas en [pruebas de contratos](tests/test_tools_contract.py) y
[del grafo](tests/test_graph_flows.py).

### Conocimiento y abstención

Construí la KB con cuatro documentos: negociación, medios de pago, escalamiento y FAQ. Fragmento
por sección con ID estable y preservo título, versión y vigencia. Trabajo con las **35 secciones
disponibles**; no inventé contenido para alcanzar los 250–350 chunks que preveía mi plan.

Uso embeddings `text-embedding-3-large` y búsqueda híbrida: similitud vectorial más búsqueda léxica
de Postgres (`ts_rank_cd`, no BM25 exacto), fusionadas con RRF. Puedo aplicar Cohere `rerank-v3.5`
y reproducir los benchmarks con cachés guardadas ([reporte](evals/reports/retrieval.md)).

En el camino generativo recupero hasta diez secciones candidatas, pido citas textuales, verifico
que existan y compruebo semánticamente que respalden la respuesta a esa pregunta. Si no encuentro
sustento, me abstengo: derivo en alto riesgo y ofrezco asistencia en bajo riesgo. Prioricé recuperar
evidencia y verificar la respuesta porque un umbral global de similitud también descartaba preguntas
respondibles.


### Límites que conservo visibles

Los `xfail` pertenecen a F5 y F7; los `skipped` necesitan Postgres. En el judge, el rechazo de tonos
incorrectos tiene TNR **0,562**: ampliaré las etiquetas de respuestas generadas antes de usarlo como
señal fuerte de calidad. Con 0/33 escapes de guardrails, mi cota superior unilateral del 95 % sigue
siendo aproximadamente **8,7 %**; no interpreto cero observaciones como ausencia de riesgo.

**El fingerprint de prompt no identifica la configuración que produjo un resultado.** Hoy obtuve
corridas con el mismo `bf5d9dec6189f4fd` y gates opuestos, porque entre ellas sólo cambió el
esfuerzo de razonamiento de una tarea. El reporte ya registra la concurrencia; todavía debe registrar
modelos, esfuerzo por tarea y timeout para que dos corridas sean distinguibles.


### Cómo investigaría "el costo subió 4× y la respuesta supera los 4 s"

1. **Fijaría una comparación equivalente.** Mismo tipo de conversación y período, separando costo por
   conversación iniciada y resuelta, latencia total y primera respuesta útil.
2. **Descompondría por etapa**, como hice arriba: modelo, tokens de entrada/salida, caché, reintentos,
   número de llamadas, retrieval y esperas. Buscaría cambios de prompt, corpus, configuración y carga
   antes de atribuir la regresión al proveedor.
3. **Probaría una hipótesis por vez** con criterio de reversión fijado antes de correr el experimento.
   Las cinco causas más probables, con su señal en la traza:
   - **Prompt caching roto** → `cached_tokens` cae a ~0. Fix: estabilizar el prefijo estático y reordenar
     dinámico al final.
   - **Contexto creció** → `input_tokens` subiendo turno a turno. Fix: compactación, top-k menor.
   - **Más llamadas LLM** → `steps/turn` sube. Fix: revisar regeneración y ruteo ambiguo.
   - **Modelo/reasoning cambió** → más tokens de razonamiento. Fix: pinear por entorno, validar con
     suite antes de cambiar.
   - **Saturación del proveedor** → p95 sube sin cambio de código, 429. Fix: backpressure y fallback.
4. **Validaría el resultado completo.** Repetiría gates y carga con caché fría y caliente, publicaría
   costo y percentiles antes/después, y desplegaría gradualmente con reversión.


## Casos de clientes y evidencia

Uso datos ficticios de [mis fixtures](mock_api/fixtures/). Elegí cuatro clientes porque ejercitan
decisiones distintas: gestionar, escalar por política, frenar por identidad y reconocer que no hay
deuda vigente. Presento los mensajes como **guiones de prueba**; los videos están pendientes.

| Cliente | Situación | Qué pruebo | Qué verifico |
|---|---|---|---|
| `CUST-00125` | mora media, ARS 184.500, 63 días | "¿Cuánto debo?" → "¿Qué opciones tengo?" → "Quiero la opción de 3 cuotas" → "Prefiero transferencia" → "Sí, confirmo" | Consulto datos reales, ofrezco condiciones permitidas, respeto el medio elegido y registro **un solo** acuerdo. Verifico ARS 61.500 por cuota, sin inventar importes. |
| `CUST-00212` | prejudicial, 190 días, dos planes incumplidos | "No puedo pagar todo, ¿me armás un plan?" | Reconozco el motivo de derivación y no intento cerrar un acuerdo ni prometer una excepción. |
| `CUST-00377` | identidad sin verificar | "Quiero pagar en cuotas" | Derivo para validar identidad y no cierro acuerdo, aunque el backend liste alternativas. |
| `CUST-00450` | sin deuda vigente | "¿Cuánto debo?" | Informo ausencia de deuda, sin inventar importes ni confundirlo con un cliente inexistente. |


Acá les pongo algunos casos para probar si copian y pegan los textos que dice "Cliente:" pueden ir probando los mismos flujos

## `CUST-00450`

Diego Sosa

Cliente sin deuda vigente.

### Caso - Consulta: 

```bash
- Agente: Soy un asistente virtual de cobranzas. La conversación se registra para gestionar tu consulta.

- Cliente: Cuánto debo?

- Agente: No registrás deuda vigente: tu último pago, de $74.200, quedó acreditado el 08/09/2026. ¿Puedo ayudarte con algo más?

- Cliente: no gracias

- Agente: De nada. Que tengas un buen día.
``` 

<video src="docs/assets/CUST00450-cliente-sin-deuda.mov" controls="controls" width="100%"></video>

## `CUST-00125`

María Gómez 

Cliente con mora media, tres vencimientos impagos. Probá saldo, opciones, un acuerdo con confirmación, quita y medios de pago.

### Caso - Negociación y acción: 

```bash
- Agente: Soy un asistente virtual de cobranzas. La conversación se registra para gestionar tu consulta.

- Cliente: No puedo pagar todo este mes. ¿Qué opciones tengo?

- Agente: Con tu situación puedo ofrecerte: (1) un pago único de $178.000; (2) 3 cuotas de $61.500 (total $184.500); (3) anticipo de $18.450 y 6 cuotas de $29.889 (total $197.784); (4) anticipo de $18.450 y 9 cuotas de $21.402 (total $211.068). La primera cuota vence el 25/09/2026. ¿Alguna te sirve?

- Cliente: la 1

- Agente: Listo, quedó registrado el compromiso N° AGR-F303F704. Si necesitás modificarlo, escribinos antes del primer vencimiento.

- Cliente: gracias

- Agente: De nada. ¿Te puedo ayudar con algo más?
```
### Registro de compromiso de pago:

<video src="docs/assets/CUST-00125-registro-compromiso-de-pago.mov" controls="controls" width="100%"></video>

### Otras consultas de Maria donde se le avisa que tiene regisrado un compromiso de pago

### Caso - Consulta

```bash
- Agente: Soy un asistente virtual de cobranzas. La conversación se registra para gestionar tu consulta.

- Cliente: ¿Cuánto debo?

- Agente: Según el sistema, al día de hoy tenés un saldo de $184.500, con 3 períodos vencidos y 63 días de atraso. Ya tenés un acuerdo de pago activo, así que no hace falta armar otro plan. Si necesitás modificarlo, te puedo derivar con un asesor.

- Cliente: puedo pagar con las cuotas de mi tarjeta?

- Agente: No se aceptan cuotas de la tarjeta del banco para planes; los planes son de la empresa y la tarjeta se usa en un pago. Los medios habilitados incluyen la tarjeta de crédito (Visa o Mastercard) del titular, en un pago. [PAY-MET-003] [PAY-MET-001]
```

<video src="docs/assets/CUST-00125-compromiso-pago-activo-mas-consulta.mov" controls="controls" width="100%"></video>

### Caso - Consulta

```bash
- Agente: Soy un asistente virtual de cobranzas. La conversación se registra para gestionar tu consulta.

- Cliente: ¿Puedo tener dos planes al mismo tiempo?

- Agente: No. Sólo se admite un acuerdo activo por cliente. [FAQ-006]

- Cliente: ¿Cuánto debo?

- Agente: Según el sistema, al día de hoy tenés un saldo de $184.500, con 3 períodos vencidos y 63 días de atraso. Ya tenés un acuerdo de pago activo, así que no hace falta armar otro plan. Si necesitás modificarlo, te puedo derivar con un asesor.
```

<video src="docs/assets/CUST-00125-con-compromiso-de-pago-activo.mov" controls="controls" width="100%"></video>


`CUST-00377`

Luciana Paz

Cliente con mora temprana con identidad sin verificar. Un plan requiere validar identidad con un asesor.

### Caso - Negociación y Escalamiento::

```bash
- Agente: Soy un asistente virtual de cobranzas. La conversación se registra para gestionar tu consulta.

- Cliente: No puedo pagar todo este mes. ¿Qué opciones tengo?

- Agente: Para avanzar con un plan, un asesor tiene que validar tu identidad. Ya te derivé para que lo revise con vos.

- Cliente: bueno

- Agente: Tu caso ya lo tiene un asesor. ¿Querés consultar tu saldo o alguna política?
```

<video src="docs/assets/CUST00377cliente-mora-temprana.mov" controls="controls" width="100%"></video>

`CUST-00212`

Jorge Ferreyra

Cliente pre-judicial. Un pedido de plan se deriva a un operador.

### Caso - Ambigüedad y Escalamiento:

```bash
- Agente: Soy un asistente virtual de cobranzas. La conversación se registra para gestionar tu consulta.

- Cliente: Quiero pagar lo que pueda

- Agente: Para armarte algo concreto necesito un dato: ¿cuánto podrías pagar este mes?

- Cliente: 20000

- Agente: Por la situación de tu cuenta, un plan de pago lo tiene que evaluar un asesor del equipo. Ya te derivé para que revise las alternativas con vos.

- Cliente: bueno

- Agente: Tu caso ya lo tiene un asesor. ¿Querés consultar tu saldo o alguna política?
```

<video src="docs/assets/CUST-00212-cliente-prejudicial.mov" controls="controls" width="100%"></video>

### Caso - Fuera de contexto:

```bash
- Agente: Soy un asistente virtual de cobranzas. La conversación se registra para gestionar tu consulta.

- Cliente: ¿Quién va a ganar el Mundial?

- Agente: Te puedo ayudar sólo con tu cuenta y las opciones de pago. ¿Seguimos con eso?
```

<video src="docs/assets/caso-fuera-de-contexto.mov" controls="controls" width="100%"></video>


Los seis comportamientos del challenge, con conversaciones independientes:

| Qué pruebo | Qué verifico |
|---|---|
| "Quiero pagar lo que pueda" | Pido precisión y no interpreto ambigüedad como confirmación. |
| "¿Quién va a ganar el Mundial?" | Redirijo al alcance de cobranzas, sin moralizar. |
| "Quiero hablar con una persona" | Registro la derivación sin insistir en negociar. |

También cubro preguntas durante la confirmación, negativas, ofertas vencidas, cambios de saldo,
fallas de tools, resultados inciertos e intentos de injection en [mis casos de
evaluación](evals/cases/) y [mis invariantes](tests/test_invariants.py).


## Cómo pienso llevarlo a producción

Pense el proyecto desde 0 para produccion, ya tiene las bases.

Me queda completar la Fase 5 antes de conectar datos reales. La Fase 6 ya está parcial —la
instrumentación (OTel, p95 por nodo/tarea, costo por conversación) corre hoy y es la que uso para
reportar latencia y costo en este README— y la usaré para validar un piloto con carga y fallos
controlados, que es lo que todavía falta. Mediré p50/p95/p99, tiempo hasta la primera cláusula útil,
errores, reintentos, espera de pools y locks, tokens por tarea, aciertos de caché y costo por
conversación resuelta bajo esa carga. Para negocio mediré acuerdos válidos, derivaciones correctas,
abandono y pagos efectivamente acreditados; **no equipararé un compromiso registrado con dinero
cobrado**.

Para escalar mantendré réplicas de API con estado compartido, límites por proveedor, admisión
controlada y presupuestos de tiempo. Probaré reconexiones, recuperación de checkpoints,
reconciliación de escrituras inciertas, backups y caídas de dependencias. Completaré autenticación
real con rotación y revocación, redacción de PII, cifrado, permisos de auditoría y retención. Hoy el
HMAC de auditoría me permite detectar alteraciones de contenido; no sustituye cifrado ni un
almacenamiento inmutable.

No prioricé fine-tuning ni multiagente: primero necesito demostrar un beneficio frente al flujo
actual con datos propios.

## Qué cambiaría para Voice AI / Realtime

Reutilizaría reglas, tools y protocolo de acuerdos; agregaría una capa de sesión de audio **STT →
agente → TTS**, con detección de turnos, streaming y control de interrupciones. Cambiaría mensajes
largos por intervenciones breves y mediría desde el fin del habla hasta el primer audio útil.
Definiría presupuestos separados para transcripción, decisión, tools y síntesis, con degradación y
transferencia si se exceden.

Ante un barge-in cortaría la reproducción y descartaría la respuesta pendiente, sin asumir que pude
cancelar una escritura ya enviada. Para errores de STT repetiría y validaría cifras; usaría DTMF para
confirmaciones críticas y **no aceptaría un "sí" transcripto como única evidencia**. Implementaría
niveles de autenticación antes de revelar deuda, continuidad de borradores al reconectar y
transferencia con resumen, estado y referencia de operación.

Todavía no implementé esa lógica de canal ni el audio de punta a punta. La postergué para cerrar
primero la seguridad y la calidad de chat; no considero que exponer SSE resuelva voz.

## Cómo cubro los entregables

| Entregable del challenge | Dónde se encuentra en este repositorio |
|---|---|
| **Repositorio con código fuente** | Código modular en [`app/`](app/), [`mock_api/`](mock_api/), [`config/`](config/), [`migrations/`](migrations/) y suite en [`tests/`](tests/). |
| **README con instrucciones de ejecución** | Sección [Cómo ejecuto y reviso la solución](#cómo-ejecuto-y-reviso-la-solución) con flujos para local rápido (`make run`), Docker (`make up`) y pruebas offline. |
| **Arquitectura y decisiones técnicas** | Sección [Arquitectura y decisiones](#arquitectura-y-decisiones) y registros formales en [`docs/decisions/`](docs/decisions/) (ADR-009 a ADR-012). |
| **Ejemplos de conversaciones y pruebas** | Casos documentados por cliente con videos interactivos, fixtures en [`mock_api/fixtures/`](mock_api/fixtures/) y suites en [`evals/cases/`](evals/cases/). |
| **Estrategia de evaluación** | Suites en capas (`make eval`, `make eval-heldout`, `make eval-blind`, `make eval-live`), calibración de judge y guardrails en [Evaluación y resultados](#evaluación-y-resultados). |
| **Diagrama de arquitectura o flujo** | Diagrama de flujo Mermaid que modela el ciclo de vida del turno y validaciones de seguridad. |
| **Partes no implementadas y justificación** | Análisis de priorización técnica y especificaciones ejecutables (`xfail`) para F5, F6 y F7 en [Por qué no prioricé F5, F6 y F7](#por-qué-no-prioricé-f5-f6-y-f7-y-cómo-las-resolvería). |