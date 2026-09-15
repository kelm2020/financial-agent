# Respondibilidad de políticas — medición end-to-end

Fecha: 2026-09-14. Se mide lo que lee el cliente, a través del agente real:

- router determinista, con el router y el clasificador de guardas con modelo;
- agente `gpt-5-nano` (`OPENAI_AGENT_MODEL`);
- embeddings `text-embedding-3-large`;
- reranker `rerank-v3.5`;
- índice en memoria de `kb/`.

Se reproduce con `make eval-answerability`, que usa la red y las claves de OpenAI y Cohere.

## Protocolo

- **Split** (`evals/answerability.yaml`):
  - 35 positivas, una por sección de la KB, escritas sólo a partir de los títulos antes de la
    primera corrida;
  - 15 negativas plausibles que la KB no responde.
- **Ejecución:** una conversación nueva por pregunta, con el cliente CUST-00125.
- **Criterio:**
  - una positiva es correcta si la respuesta cita al menos una de las secciones esperadas;
  - una negativa es correcta si la respuesta no cita ninguna sección.
- **Anti-contaminación:** es la misma regla que en `evals/blind`. Cada frase que motivó un cambio
  pasó a un test unitario (`tests/test_policy_routing.py`, `tests/test_policy_answers.py`) y se
  reemplazó en el split antes de volver a medir: 11 frases después de la corrida 1, 1 después de la
  corrida 2 y 3 después de la corrida 4. Los demás cambios son estructurales y no dependen de ninguna
  frase. Tras varias corridas, el split ya no es completamente ciego.

## Resultados

| Corrida | Cambio previo | Modelo | Positivas con cita correcta | Negativas sin respuesta citada | Citas incorrectas |
|---|---|---|---:|---:|---:|
| 1 | Toda respuesta de política con modelo: máximo recall más citas verificadas | nano | 16/35 | 11/15 | 7 |
| 2 | Ruteo de preguntas de política; verificación de toda respuesta generada; rechazo de oraciones que repiten la pregunta | nano | 19/35 | 12/15 | 8 |
| 3 | Búsqueda sin filtro de tema; regeneración sin presupuesto cae al extracto | nano | 26/35 | 11/15 | 5 |
| 4 | Extracto cuando el modelo se abstiene y la evidencia supera el gate; títulos en la verificación | nano | 26/35 | 12/15 | 4 |
| **Final** | **Vista para el cliente única; foco en la sección principal y sus referencias; afirmaciones sin respaldo descartadas; intenciones definidas en el router con modelo** | **nano** | **27/35** | **13/15** | **5** |
| Final | Igual, con `OPENAI_AGENT_MODEL=gpt-5-mini` | mini | 26/35 | 13/15 | 4 |

Con el código final, `gpt-5-mini` no supera a `gpt-5-nano`: las diferencias que se veían con código
anterior (14/15 negativas contra 11/15) venían del camino de respuesta, no del modelo. Se mantiene
`gpt-5-nano`.

## Diseño que surgió de las corridas y del chat local

- **Una sola vista para el cliente de cada sección** (`customer_view`). La leen el modelo, el
  extracto y el verificador. Se quitan tres cosas:
  - las oraciones dirigidas al agente;
  - las referencias internas entre paréntesis, como "(POL-NEG-006)";
  - la autorreferencia "este documento", que pasa a "estas políticas".

  El verificador acepta además el texto original, así que una cita literal nunca produce un falso
  bloqueo.
- **Foco en la sección que responde.** La sección principal es la mejor rankeada entre las que citó
  el modelo, y otra sección se mantiene sólo si la principal la referencia (FAQ-002 → PAY-MET-002).
  El enlace inverso quedó afuera: ESC-001 lista "Pide una excepción a las políticas (POL-NEG-009)"
  y ese ítem, fuera de contexto, confundía.
- **Afirmaciones descartadas, no bloqueadas.** Una oración con cita literal que no conserva al menos
  dos tercios de sus términos en la sección citada se quita de la respuesta; por ejemplo, "Podés
  cambiar el medio de pago…" citada como FAQ-003. Como bloqueo duro, esa regla rechazó una salida
  correcta del split held-out de guardrails. Por eso quedó como filtro del agente y no del
  verificador.
- **Observabilidad:** el evento `claims_trimmed` registra cuántas afirmaciones se descartaron por ser
  internas, por falta de respaldo o por pertenecer a otra sección.
- **Router con modelo:** se definieron las intenciones. Una pregunta sobre cómo funcionan las reglas
  es `consulta_general` aunque mencione la deuda, un plan o una oferta.
- **Chequeo de pertinencia léxica, probado y descartado:** la regla "la sección citada comparte un
  término distintivo de la pregunta" rechazaba respuestas correctas parafraseadas; por ejemplo,
  "no llego a pagar" frente a "impaga".

## Fallas de la corrida final (nano)

- **Deriva en lugar de responder (Q-01, Q-02):** son preguntas sobre derivación y vulnerabilidad,
  que activan la derivación. Se prioriza no perder un pedido real.
- **Ruteo:**
  - Q-18: el router con modelo la manda a los vencimientos del cliente.
  - Q-24: "reducción de intereses" no está en la KB ni en el split dev.
  - Q-27: la manda a las opciones.
- **Sección distinta de la etiquetada:**
  - Q-11 cita PAY-MET-004 en lugar de FAQ-005;
  - Q-13 cita FAQ-002 y PAY-MET-002 en lugar de FAQ-007;
  - Q-20 cita POL-NEG-008 y FAQ-004 en lugar de FAQ-014.
- **Negativas con respuesta (U-04, U-09):** citan texto real relacionado que no responde lo que se
  preguntó.

## Riesgo residual

- **Qué verifica la verificación:** es extractiva. Controla cuatro cosas:
  - la cita es literal;
  - cada oración visible está respaldada;
  - ninguna oración toma de la pregunta un término ausente en su sección;
  - cada afirmación conserva la mayoría de sus términos en la sección que cita.
- **Qué no verifica:** que la sección responda exactamente la pregunta.
- **Mitigación sin probar:** un chequeo de pertinencia con una segunda llamada al modelo, que
  suma costo y latencia a cada respuesta de política.
- **Sin modelo:** el agente sigue usando el gate calibrado y el extracto literal
  (`evals/reports/retrieval.md`).
