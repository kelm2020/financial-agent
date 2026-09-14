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
  reemplazó en el split antes de volver a medir: 11 frases después de la corrida 1 y 1 después de la
  corrida 2. Los cambios de la corrida 3 son estructurales y no dependen de ninguna frase. Tras tres
  corridas, el split ya no es completamente ciego.

## Resultados

| Corrida | Cambio previo | Positivas con cita correcta | Negativas sin respuesta citada | Citas incorrectas |
|---|---|---:|---:|---:|
| 1 | Toda respuesta de política con modelo: máximo recall más citas verificadas | 16/35 | 11/15 | 7 |
| 2 | Ruteo de preguntas de política; verificación de toda respuesta generada; rechazo de oraciones que repiten la pregunta | 19/35 | 12/15 | 8 |
| **3** | **Búsqueda sin filtro de tema; regeneración sin presupuesto cae al extracto** | **26/35** | **11/15** | **5** |

Hallazgos de las corridas 1 y 2 que motivaron cambios:

- **Oraciones con una cita real que no las respalda:** "La comisión del asesor es del 10 % del
  saldo total. [FAQ-001]" pasó porque las respuestas de riesgo bajo no se verificaban. Ahora se
  verifica toda respuesta del modelo, y se rechaza la oración que toma de la pregunta un término que
  la sección citada no contiene.
- **El tema como filtro ocultaba la sección correcta:** el router con modelo clasificó "¿en qué
  horario atienden los asesores?" como `faq`, y ESC-004 nunca se recuperó. Con modelo, la búsqueda
  cubre todas las secciones y el tema sólo fija el riesgo.
- **Presupuesto de 3 llamadas por turno:** el clasificador, el router, la respuesta y una
  regeneración suman 4, así que una pregunta de política terminaba derivada. Ahora la regeneración
  sin presupuesto cae al extracto o a la abstención.

## Fallas de la corrida 3

- **Deriva en lugar de responder (Q-01, Q-02, Q-21):** son preguntas sobre derivación,
  vulnerabilidad o reclamos, y activan la derivación. Se prioriza no perder un pedido real.
- **El modelo se abstiene con la sección correcta recuperada:**
  - Q-05 (ESC-005), Q-09 (FAQ-003) y Q-13 (FAQ-007);
  - en Q-17 (FAQ-011) la verificación rechazó las dos respuestas.
- **Ruteo (Q-24):** "reducción de intereses" va a la composición del saldo. "Reducción" no aparece
  en la KB ni en el split dev, así que no se agrega como vocabulario.
- **Sección incorrecta (Q-20):** una pregunta sobre dar de baja un plan aceptado cita POL-NEG-008 y
  FAQ-004 en lugar de FAQ-014.
- **Negativas con respuesta (U-03, U-04, U-09, U-15):** citan texto real que no responde la
  pregunta. No hay cifras inventadas, pero U-03 es engañosa: "Sí, desde el 10 % del saldo total"
  ante "¿puedo deducir estos pagos en mi declaración de impuestos?".

## Riesgo residual

- **Qué verifica la verificación:** es extractiva. Controla tres cosas:
  - la cita es literal;
  - cada oración visible está respaldada por un claim;
  - ninguna oración toma de la pregunta un término ausente en su sección.
- **Qué no verifica:** que la sección responda la pregunta (U-03).
- **Mitigaciones posibles**, las dos con costo:
  - un modelo del agente más capaz para las respuestas de política;
  - un chequeo de pertinencia con una segunda llamada al modelo.
- **Sin modelo:** el agente sigue usando el gate calibrado y el extracto literal
  (`evals/reports/retrieval.md`).
