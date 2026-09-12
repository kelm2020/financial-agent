---
doc_id: ESC
titulo: Criterios de escalamiento a un operador
version: 1.0.0
status: approved
effective_from: 2026-01-01
effective_to: null
audiencia: [agente, operador]
---

## ESC-001 · Derivación inmediata

Se deriva sin negociar ni insistir cuando:

1. El cliente pide hablar con una persona.
2. Desconoce la deuda o presenta un reclamo.
3. Hay indicios de fraude o la identidad no está verificada.
4. Menciona un abogado, una demanda o una acción judicial.
5. Pide una excepción a las políticas (POL-NEG-009).
6. El segmento es prejudicial, o registra dos o más planes incumplidos.
7. Faltan datos o son inconsistentes y no se pueden obtener del sistema.
8. Una consulta al sistema falla después de los reintentos previstos.
9. El resultado de un registro de acuerdo queda incierto (ESC-003).
10. La interacción se vuelve abusiva o amenazante.

## ESC-002 · Situación declarada de vulnerabilidad

Si el cliente declara una situación de vulnerabilidad económica o personal que le impide
afrontar la deuda, el agente **no negocia**: reconoce lo que la persona dijo en una oración,
no pide detalles, no repite lo dicho y deriva con prioridad. La marca de vulnerabilidad se
registra como bandera, sin transcribir el relato.

## ESC-003 · Fallas técnicas y resultados inciertos

Si un registro de acuerdo se envía y la respuesta no llega o es ambigua, el agente **no
afirma ni niega** que el acuerdo quedó registrado. Informa que no puede confirmarlo todavía
y deriva con el identificador de la operación, para que un operador reconcilie.

Nunca se le dice a un cliente que algo quedó registrado sin confirmación del sistema.

## ESC-004 · Horarios y tiempos

Atención de operadores: lunes a viernes de 9 a 18 y sábados de 9 a 13, hora de Argentina.
Fuera de ese horario la derivación queda registrada y un operador retoma el próximo día
hábil. El agente **no promete tiempos de espera ni de respuesta**.

## ESC-005 · Trato y límites de contacto

No se amenaza con consecuencias que no estén previstas en el contrato. No se menciona a
terceros ni se habla de la deuda con quien no sea el titular. No se contacta fuera de la
franja de 8 a 21. No se presiona, no se usa urgencia artificial y no se insiste después de
una negativa clara.

Una sola derivación por sesión: si el cliente vuelve a pedirlo, se informa el mismo ticket.

## ESC-006 · Datos personales

El agente sólo accede a datos del titular autenticado de la conversación, nunca de otro
cliente. No solicita claves, códigos de seguridad ni el número completo de una tarjeta.
El cliente puede ejercer sus derechos de acceso y rectificación sobre sus datos; ese pedido
se deriva al canal correspondiente.
