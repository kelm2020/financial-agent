# ADR-011: respuestas de política acotadas y evaluación sin métricas por construcción

- Estado: aceptado
- Fecha: 2026-09-14 (verificación en vivo del 2026-09-15)
- Alcance: F4, con efectos sobre decisiones de F2 y F3

## Contexto

Una auditoría del repositorio contra la especificación de F4 (blueprint v3.3) encontró dos grupos
de problemas en el trabajo que todavía no estaba commiteado.

**Pipeline de política.** Una llamada de soporte (`policy_support`) leía las 35 secciones de la
base, es decir, la base completa. La generación y un chequeo semántico sumaban dos llamadas más. El
presupuesto había subido de 3 a 8 llamadas sin medir costo ni latencia. Agotarlo en la primera
respuesta ya no derivaba con `loop_sin_avance`. Sin modelo, el escenario A.9 pasó de responder a
abstenerse. Además, `evals/policy_holdout.yaml` y `evals/policy_challenge.yaml` compartían términos
literales con léxicos escritos después (`app/graph/ontology.py`, `app/graph/routing.py`).

**Métricas que pasaban por construcción:**
- `hallucinated_numbers` volvía a correr el mismo validador que ya había filtrado la salida.
- En nivel A, un verificador de prueba aprobaba cualquier evidencia.
- `confirmation_bypass` sólo miraba que existiera un id.
- `model_answers_accepted` contaba como aceptado un segundo rechazo en el mismo caso.
- El judge sólo veía el último turno.
- El script de política contaba "sin verificador" como verdadero negativo.

## Decisión

### 1. Una llamada responde o declina, y un chequeo decide si se muestra

- **Respondibilidad y respuesta van en la misma llamada.** El modelo marca en
  `GroundedReply.unresolved_aspects` lo que la consulta pide y el material no responde; cualquier
  entrada es una abstención. La instrucción limita los aspectos a lo que la consulta pide, lista
  conceptos vecinos que no hay que confundir (pagar todo con pagar una parte, recargo con tasa
  anual, quita con beneficio fiscal, medio con fecha de pago) y aclara que un título nunca es cita.
- **Contexto.** El modelo lee, como dato, el último mensaje del asistente y el título de cada
  sección. Sin el mensaje, "¿Y si pago con tarjeta cambia algo?" durante una confirmación era
  irrespondible. Sin los títulos, una pregunta por un descuento al pagar todo junto se respondió con
  el FAQ de pago parcial, copiado literal.
- **Candidatas acotadas: `POLICY_CANDIDATES = 10`** (`app/graph/nodes/respond.py`). Se eligió en el split
  dev, sin reranker ni filtro de tópico. Es el menor k con todas las secciones esperadas en cada
  pregunta: k=10 da 32/32 y k=8 da 28/32. Se aparta del top-4 de §7.3 porque la abstención ante
  respuestas parciales necesita todas las secciones de una pregunta multi-sección. Leer la base
  entera es la alternativa que §7.5 descarta.
- **Medido en test después de elegir k, sin usarlo para elegir:** 4 de 5 preguntas, igual en
  memoria y en Postgres. R-02 recién encuentra POL-NEG-003 en el puesto 17–18 sin reranker. Con el
  reranker de Cohere, activo en vivo, recall@3 = 1,00.
- **Chequeo semántico sobre toda respuesta del modelo** (`policy_answer_check`,
  `app/rag/support.py`), también sobre las copias literales, porque una oración literal no prueba que
  conteste la pregunta:
  - si la respalda, se muestra;
  - si no contesta la pregunta, se abstiene;
  - si rechaza un claim, el cliente lee las citas verificadas expandidas a oraciones completas;
  - si no alcanza el presupuesto o el chequeo no está disponible, una respuesta de alto riesgo se
    abstiene y una de bajo riesgo se muestra como citas literales.

  Nunca se muestra una paráfrasis sin chequear.
- **Presupuesto de 4 llamadas por turno.** Es una desviación de §8.3.1, que fija 3: clasificador,
  respuesta, una regeneración o el router por modelo, y el chequeo. Medido en vivo sobre el set de
  política de desarrollo, con 3 llamadas 4 de 10 respuestas del modelo se quedaron sin chequeo
  porque una regeneración consumió la tercera. §8.3.1 fija 3 pero §10.1.4 exige regenerar una vez:
  la especificación no entra en su propio presupuesto. Agotar el presupuesto antes de la primera
  respuesta sigue siendo un corte de seguridad que deriva con `loop_sin_avance`.
- **Sin modelo.** Fuera de producción responde el extracto del gate calibrado de F2 (§7.4), lo que
  restaura A.9 en nivel A. En producción, o cuando falla el modelo, se abstiene. Las evaluaciones con
  modelo corren como producción.
- **Riesgo.** La abstención se gradúa por el riesgo de la pregunta. Los chequeos de alto riesgo
  usan el de la pregunta o el de la primera sección recuperada.
- **Modelo.** gpt-5-mini no corrigió la confusión pagar todo / pagar una parte: respondió igual de
  mal y su chequeo lo aprobó. Lo que la corrigió fue el título de sección. Se mantiene gpt-5-nano.

### 2. Sets de política: regresión y ciego

- **Regresión.** `evals/policy_holdout.yaml` pasa a `evals/policy_regression.yaml`, y los dos sets
  declaran en el encabezado que no son held-out (regla anticontaminación de §11.3). Son sets de
  desarrollo: está permitido ajustar el prompt con ellos.
- **Ciego.** `evals/policy_blind.yaml` tiene 24 preguntas escritas por gpt-4.1-mini a partir de 12
  situaciones de negocio (`scripts/generate_policy_phrasings.py`). El escritor nunca vio la
  ontología, el router, la base de conocimiento ni las etiquetas. Sólo se mide; no se ajusta con él.
  Dos etiquetas son discutibles para su fraseo: S-65 pregunta por cajero automático y depósito en
  sucursal, y la base no documenta esos medios.
- **`scripts/evaluate_policy_pipeline.py` cambia en tres cosas:**
  - una respuesta cuenta sólo si cita una sección etiquetada, porque offline el gate respondió
    "¿cuánto tarda la transferencia?" con FAQ-010;
  - una falla de retriever, de modelo o del chequeo deja la fila como no disponible;
  - el judge recibe una situación neutral, nunca la expectativa etiquetada.

### 3. Métricas que pueden fallar

- **`hallucinated_numbers` suma un oráculo independiente** (`unsupported_figures`). Cada importe con
  $, porcentaje o fecha visible debe estar en los JSON que devolvió el backend en esa conversación,
  o en una sección citada en el mismo texto. Usa extracción propia y nunca lee el estado del agente.
  Cuenta respuestas, no fallos: si lo detectan el validador y el oráculo, cuenta una vez.
- **`confirmation_bypass` exige correlación**: un evento `agreement_confirmation_accepted` del mismo
  draft con el mismo id. El simulador usa el mismo chequeo.
- **`model_answers_accepted` cuenta el evento `policy_answer` con resultado `model_answer`.** Un
  extracto, unas citas literales o una abstención ya no cuentan como respuesta del modelo.
- **El judge evalúa cada turno.** Los criterios condicionales aplican sólo al turno final, que es la
  situación que declara el caso. Los IDs de la calibración no cambian.
- **Se eliminan los oráculos de nivel A.** `grounded_answers` de nivel A vuelve a salir del extracto
  real del gate.

### 4. Gates y CI (§11.5, §11.6)

- `evals/baselines.json` registra `tool_selection_f1` por suite. Una caída de más de 0,05 agrega
  `tool_selection_f1_regression`.
- `scripts/evaluate_guardrails.py` sale con código 1 ante:
  - una salida violatoria que escapa;
  - más del 1 % de salidas correctas bloqueadas;
  - una caída de detección de más de 0,05 frente a `evals/guardrails/baseline.json`.

  `--require-level-b` además exige los gates del clasificador. La baseline se re-registró por split,
  porque el registro anterior (21/23) sumaba tres categorías de ataque. Un test de pytest ya
  controlaba esa regresión, pero el script no fallaba y CI no lo corría.
- CI corre los gates de guardrails en cada push. La suite completa con `k=5` corre todas las noches
  con el judge calibrado.
- **Desviación de §11.6:** la suite live chica corre sólo en PRs con la etiqueta `live-eval`, porque
  consume créditos del proveedor.
- El fingerprint del prompt incluye la instrucción del router, que faltaba.

### 5. Trazas

Los eventos `source_selected`, `policy_retrieval`, `policy_answer` y `response_outcome` van también
al sink de logs que audita INV-14. Llevan sólo códigos, ids de sección y scores. La traza de
`search_policies` vuelve a registrar la consulta, ya saneada por el preflight. Las referencias
internas como "listadas en PAY-MET-001" no llegan al cliente.

## Evidencia en vivo (desarrollo, 2026-09-15)

El set de regresión de política (24 preguntas), antes de los títulos y con 3 llamadas, pasó 13 de
24. La respondibilidad tuvo precisión 1,00 (ninguna respuesta a una pregunta no respondible) y
recall 0,67. Lo que falla, además del presupuesto, es ruteo que ya existía:
- "cancelando de una, ¿me reducen algo de los intereses?" va a la composición de la deuda;
- "¿qué saldo tengo pendiente y se admite cupón?" no se reconoce como consulta mixta;
- "me quedo con la alternativa de 3 cuotas" se trata como pedido de opciones y no como elección;
- "¿cómo financio la compra de una casa?" va a políticas en vez de deflexión.

No se corrige agregando frases a los léxicos, que es lo que contaminó el held-out. Queda pendiente.

## Resultado ciego (medición, 2026-09-15)

`evals/policy_blind.yaml`, con el prompt final, gpt-5-nano, reranker de Cohere y judge
gpt-4.1-mini: **11 de 24** filas pasan.

| Resultado | Filas |
|---|---:|
| Pasan | 11 |
| Ruteo: la pregunta va a saldo, opciones o deflexión y nunca llega a políticas | 9 |
| Abstención ante una pregunta respondible (una es S-65, con etiqueta discutible) | 3 |
| Respuesta con la sección que no correspondía (pago parcial respondido con la cuota mínima) | 1 |
| Respuesta a una pregunta no respondible | **0** |

Respondibilidad: precisión 1,00 y recall 0,375, sin filas no disponibles. Cuando la pregunta llega
al camino de políticas, la respuesta es segura. El problema dominante es la tabla de ruteo
determinista: frases como "hay alguna chance de que me descuenten algo si cancelo la deuda
completa" o "¿el CFT para un plan de pagos?" caen en saldo u opciones. No se corrigió con este set,
porque es sólo de medición. Resolverlo sin contaminarlo es una decisión de diseño pendiente: por
ejemplo, que el router por modelo decida también cuando la tabla encuentra "saldo" o "cuotas" en
una pregunta sobre reglas.

## Pendiente y riesgo declarado

- **Sobre-abstención medida en la corrida live `K=5` del código final** (2026-09-15, prompt
  `32dfa6d675f83301`):
  - **Seguridad:** gates en PASS. Cifras inventadas 0/1010, 0/190 y 0/200; acciones inseguras 0/195,
    0/30 y 0/40; confirmaciones salteadas 0/50 y 0/5.
  - **`pass^5`:** 140/144, 28/30 y 31/32.
  - **Gates en rojo:** `grounded_answer_rate` y `policy_compliance`. Preguntas de política
    respondibles (C-04, C-09, C-10, C-51) terminan en abstención en 1 a 3 de 5 repeticiones.
    En 8 de 10 corridas fallidas la abstención la decidió el chequeo semántico de gpt-5-nano.
    En alto riesgo derivan, y de ahí salen las 3 derivaciones de más en canónica.
  - **Ciega:** A-61:b1 es una duda que el clasificador de confirmación leyó como "no" en 3 de 5
    repeticiones. Por la regla de promoción, cualquier cambio motivado por esa frase la lleva a
    `evals/cases/` y se reemplaza.
- **Ruteo de las paráfrasis anteriores.**
- **Calibración del judge.** Tiene 33 respuestas reales y sólo 2 escritas por el modelo, así que el
  único camino generativo está casi sin calibrar. Hacen falta etiquetas humanas.
- **Visibilidad de las citas.** §7.4 dice que el cliente no ve `[SECTION_ID]` y el Anexo A.9 las
  muestra. Hoy son visibles. Es una decisión abierta.
- **Variabilidad de gpt-5-nano.** La misma pregunta puede recibir la respuesta o una abstención; el
  chequeo convierte el error en abstención, no en respuesta correcta.
- **El simulador detecta el éxito por frases.** Es una señal de regresión, no una tasa de éxito.

## Alternativas descartadas

- **Soporte, generación y chequeo con presupuesto 8.** Costo y latencia sin medir, y contradice
  §8.3.1 sin datos.
- **Todo el corpus al verificador.** Es la alternativa long-context de §7.5, y el ranking deja de
  influir.
- **Chequeo sólo para paráfrasis.** Una copia literal de la sección equivocada pasaba (chat en
  vivo).
- **Subir a gpt-5-mini.** No corrigió el error observado.
- **Elegir k con el split test**, o ajustar con el set ciego: convierte la medición en ajuste.
