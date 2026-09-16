# ADR-012: cierre de Fase 3–Fase 4 contra el challenge

- Estado: aceptado
- Fecha: 2026-09-15
- Alcance: Fase 3 y Fase 4. No abre Fase 5–Fase 8, que quedan diferidas (ver README)

## Contexto

Durante una revisión profunda se encontraron fallas que los tests no mostraban, porque cada uno probaba su pieza por separado:

- **SSE sin streaming.** La API esperaba el turno completo antes de emitir. El filler de una
  respuesta de política ("Dejame revisar la política, un segundo.") llegaba junto con la
  respuesta, y el tiempo hasta la primera cláusula validada era la latencia total del turno
  (§10.1.5, punto 8).
- **Sin trazabilidad de efectos.** La tabla `audit_events` existía desde la migración 0001, pero
  nada escribía en ella. Confirmaciones, acuerdos, resultados inciertos y derivaciones sólo
  quedaban en eventos del turno y en logs (punto 10: "trazabilidad de acciones ejecutadas").
- **Medio de pago impuesto.** El draft tomaba el primer medio que admitía la opción (débito
  automático). El cliente no podía elegir otro: "prefiero transferencia" no cambiaba nada y un
  "sí" registraba débito.
- **Aceptación de F3 sin medir.** `benign_deflect_rate ≤ 0,02` sobre test exige el clasificador
  real y al menos 149 benignos. No había herramienta para correrlo y el split tenía 20.
- **Lecturas en serie.** Cliente y deuda se pedían una después de la otra aunque son
  independientes.
- **Un test que dependía del reloj.** El rate limit dormía 60 ms contra una ventana de 50 ms y
  fallaba de forma intermitente bajo carga.
- **Logs apagados por las migraciones.** `migrations/env.py` llamaba a `fileConfig` con
  `disable_existing_loggers` en su valor por defecto. Toda migración ejecutada en proceso (los
  tests de integración, `scripts.initialize_database`) deshabilitaba los loggers de la aplicación.

## Decisión

### 1. El turno es una tarea propia y la API emite a medida que se valida

`ConversationAgentService.start_turn` resuelve antes del primer byte:
- la propiedad de la conversación (404);
- el rate limit (429);
- el preflight (413);
- el lock de la conversación (409).

También resuelve antes del primer byte el único status que decide el grafo: una opción
desconocida (404), que se publica antes de que `render_and_validate` emita. El grafo corre en una
tarea propia y los eventos validados salen por una cola.

- **Un cliente que se desconecta deja de leer, no corta el turno.** La escritura, su checkpoint y
  la liberación del lock terminan igual; `result()` está protegido con `asyncio.shield`.
- **El gateway vive lo que dura el turno**, no lo que dura el handler.
- **Apagado ordenado:** `drain()` espera los turnos en curso antes de cerrar el pool y el
  checkpointer.
- **Compatibilidad:** `send_message` se mantiene como el mismo turno llevado hasta el final, para
  los tests y las evaluaciones.

### 2. Auditoría durable de los efectos

`app/runtime/audit.py` escribe una fila por cada:
- escritura de acuerdo confirmada, antes de intentarla;
- resultado (creado, ya existente, incierto, rechazado);
- derivación a una persona.
