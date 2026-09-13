# ADR-009: límites del grafo, persistencia y efectos

- Estado: aceptado
- Fecha: 2026-09-13
- Alcance: F3

## Contexto

El checkpoint contiene datos financieros, historial conversacional y drafts de acuerdos. Cargarlo
antes de comprobar ownership expone información entre clientes. A la vez, una confirmación puede
ser reintentada o ejecutarse desde procesos distintos, y ningún texto del modelo puede emitirse
antes de validar su contenido completo.

## Decisión

1. La API resuelve `conversation_id + customer_id` en `conversations` antes de obtener el
   `thread_id` o invocar LangGraph. Un recurso ajeno se presenta como 404.
2. Cada ejecución toma un advisory lock de sesión por `conversation_id`. El lock se libera
   explícitamente en `finally`; el hook `reset` del pool ejecuta `pg_advisory_unlock_all()` como
   defensa adicional.
3. `AsyncPostgresSaver` recibe su propio `AsyncConnectionPool`. Su `setup()` sólo se llama desde
   bootstrap/migración o pruebas explícitas, no al iniciar cada réplica.
4. `GraphContext` contiene autoridad y dependencias no persistentes. `AgentState` contiene sólo
   datos serializables necesarios para reanudar la conversación.
5. El acuerdo usa un draft tipado y congelado. Antes del POST se releen deuda/opciones y se vuelve
   a comprobar vigencia, fingerprint, términos y política con un reloj fresco. El backend conserva
   la autoridad final mediante idempotencia.
6. `ResponsePlan` no contiene texto generado. `render_and_validate` es el único nodo que crea un
   `AIMessage` y los únicos eventos públicos son `filler` y `validated_clause` ya validados.
7. Se conservan los últimos ocho turnos. El contexto anterior se compacta en un resumen
   determinista, validado y encapsulado como texto no confiable antes de reutilizarlo.

## Consecuencias

- Una conversación se serializa entre sesiones y procesos; conversaciones distintas pueden usar
  conexiones diferentes.
- Los advisory locks de sesión ocupan una conexión durante todo el turno. El timeout del pool y
  del lock comparten un único presupuesto y se reportan como 409 `CONVERSATION_BUSY`.
- El SSE se entrega por cláusulas después de validar la respuesta completa, no token a token.
- Un outcome de escritura desconocido conserva el draft original y la clave idempotente. El turno
  siguiente reenvía exactamente el mismo request para reconciliarlo; nunca construye otro draft.
- El nivel A de guardrails no simula un clasificador. Las métricas del clasificador requieren un
  artefacto externo completo de nivel B.

## Alternativas descartadas

- Cargar el checkpoint y comprobar `state.customer_id` después: la fuga ya habría ocurrido.
- `interrupt()` para la confirmación del cliente: al reanudar puede reejecutar código anterior al
  interrupt y no resuelve preguntas intermedias.
- Emitir tokens del modelo y validarlos después: un token enviado no puede retirarse.
- Un único connection object para checkpointer y locks: reduce paralelismo y mezcla ciclos de vida.
