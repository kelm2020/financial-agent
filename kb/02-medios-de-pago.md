---
doc_id: PAY-MET
titulo: Medios de pago habilitados
version: 1.0.0
status: approved
effective_from: 2026-01-01
effective_to: null
audiencia: [agente, operador]
---

## PAY-MET-001 · Medios habilitados

- **Débito automático** sobre CBU del titular. Es el medio recomendado para planes en cuotas.
- **Transferencia bancaria** al CBU oficial de la empresa, informando el número de cuenta
  como referencia.
- **Tarjeta de crédito** (Visa o Mastercard) del titular, en **un pago**.
- **Cupón de pago en efectivo** para abonar en Rapipago o Pago Fácil.

## PAY-MET-002 · Plazos de acreditación

| Medio | Acreditación |
|---|---|
| Débito automático | mismo día |
| Transferencia bancaria | 24 horas hábiles |
| Tarjeta de crédito | 48 horas hábiles |
| Cupón en efectivo | 72 horas hábiles |

Hasta que el pago se acredita, el sistema sigue mostrando la deuda como vigente. Si el
cliente dice que ya pagó y el plazo todavía no venció, se le explica el plazo; si el plazo
ya pasó, se deriva a un operador.

## PAY-MET-003 · Medios no habilitados

No se aceptan criptomonedas, cheques de terceros, tarjeta de crédito de un tercero sin
autorización escrita, efectivo en domicilio, ni billeteras virtuales distintas de las
listadas en PAY-MET-001. Tampoco se aceptan cuotas de la tarjeta del banco: los planes
son de la empresa y la tarjeta se usa en un pago.

## PAY-MET-004 · Comprobantes

El comprobante del acuerdo y el detalle de cuotas se envían por correo electrónico a la
dirección registrada dentro de las **24 horas**. El agente puede confirmar que el envío
quedó programado; no puede adjuntar archivos ni cambiar la dirección de correo.

## PAY-MET-005 · Cambio de medio de pago

El medio de pago de un plan vigente se puede cambiar hasta **48 horas antes** del próximo
vencimiento. Si el plan ya tiene una cuota impaga, el cambio lo gestiona un operador.
