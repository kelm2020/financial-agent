---
doc_id: POL-NEG
titulo: Políticas de negociación y refinanciación
version: 1.0.0
status: approved
effective_from: 2026-01-01
effective_to: null
audiencia: [agente, operador]
---

## POL-NEG-001 · Alcance y quién puede refinanciar

Puede refinanciar el titular de la cuenta, autenticado, con deuda vencida y sin un plan de
pago activo. Un cliente no puede tener más de **un acuerdo activo** a la vez. Si ya tiene uno
vigente, el agente informa el acuerdo existente y no ofrece otro.

No se refinancia sin deuda vencida, ni a nombre de un tercero, ni a un cliente cuya identidad
no esté verificada: en ese caso se deriva a un operador.

## POL-NEG-002 · Segmentación por días de mora

El segmento se calcula sobre los días transcurridos desde el vencimiento impago más antiguo:

| Segmento | Días de mora |
|---|---|
| Mora temprana | 1 a 60 |
| Mora media | 61 a 120 |
| Mora tardía | 121 a 180 |
| Prejudicial | 181 o más |

El segmento determina el plan máximo, la quita disponible y el anticipo exigido. Los casos
**prejudiciales** no los gestiona el canal automático: se derivan siempre a un operador.

## POL-NEG-003 · Quitas de interés por segmento

La quita se aplica **únicamente sobre los intereses devengados**, nunca sobre el capital, y
**sólo en el pago único**. Un plan en cuotas no acumula quita.

| Segmento | Quita máxima de intereses |
|---|---|
| Mora temprana | 0 % |
| Mora media | 20 % |
| Mora tardía | 40 % |
| Prejudicial | requiere operador |

El agente no puede aprobar una quita mayor a la de la tabla ni aplicarla a un plan en cuotas.
Todo pedido por encima de estos valores es una excepción (POL-NEG-009).

## POL-NEG-004 · Refinanciación en cuotas y recargo por financiación

Cantidad máxima de cuotas por segmento: mora temprana **6**, mora media **9**,
mora tardía **12**. Prejudicial: requiere operador.

El recargo por financiación se aplica sobre el **monto financiado** (saldo menos anticipo):

| Cuotas | Recargo |
|---|---|
| 1 (pago único) | 0 % |
| 2 a 3 | 0 % |
| 4 a 6 | 8 % |
| 7 a 9 | 16 % |
| 10 a 12 | 24 % |

La cuota se redondea al peso y la última absorbe la diferencia.

## POL-NEG-005 · Anticipo

Se exige anticipo a partir de **4 cuotas**. Porcentaje mínimo sobre el saldo total:
mora temprana **0 %**, mora media **10 %**, mora tardía **15 %**.

Los planes de hasta 3 cuotas no requieren anticipo. El anticipo se paga junto con la
aceptación del plan y se descuenta del monto a financiar.

## POL-NEG-006 · Cuota mínima y pago parcial

La cuota no puede ser inferior a **$15.000**. Si un plan da una cuota menor, ese plan no se
ofrece: se ofrece el plan más largo cuya cuota alcance el mínimo.

Un **pago parcial** a cuenta se acepta desde el **10 % del saldo total**. El pago parcial no
suspende la gestión de cobranza, no otorga quita y no reemplaza un acuerdo.

## POL-NEG-007 · Vigencia de las ofertas y fecha de la primera cuota

Una oferta comunicada al cliente vale **48 horas**. Vencido ese plazo hay que recalcular:
el saldo puede haber cambiado.

La primera cuota vence entre **5 y 15 días corridos** desde la aceptación. No se aceptan
primeras cuotas fuera de esa ventana.

## POL-NEG-008 · Incumplimiento de un plan

Si una cuota queda impaga **10 días corridos** después de su vencimiento, el plan se da de
baja. Consecuencias: se pierde la quita aplicada, lo pagado se imputa al saldo y la deuda
vuelve al saldo original menos los pagos acreditados.

Un cliente con **dos o más planes incumplidos** no puede refinanciar por el canal automático:
se deriva a un operador (ver ESC-001).

## POL-NEG-009 · Excepciones

Cualquier condición fuera de los límites de este documento —más cuotas, más quita, menos
anticipo, cuota bajo el mínimo, cambio de fechas— es una **excepción** y sólo la puede
evaluar un operador. El agente no la aprueba, no la anticipa como probable y no sugiere
que "se puede pedir": informa que lo revisa una persona y deriva.
