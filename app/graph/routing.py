from __future__ import annotations

import re
from typing import Literal

from app.graph.ontology import concepts, mixed_request, policy_request
from app.graph.state import RouteResult
from app.guards.normalize import detection_skeleton
from app.guards.numbers_es import numbers_in_words
from app.tools.schemas import MedioPago

_OPTION = re.compile(r"\bOPT-[A-Z0-9]{2,10}\b", re.IGNORECASE)
_INSTALLMENTS = re.compile(r"\b(\d{1,2})\s+cuotas?\b", re.IGNORECASE)
# What the agent offered, by any of its names: "la alternativa de 3 cuotas" is the option of 3.
_OFFERED = r"(?:(?:opcion|alternativa|propuesta|plan) )?"
_INSTALLMENT_CHOICE = re.compile(
    rf"(?:^|\b)(?<!no )(?:quiero|elijo|tomo|prefiero|agarro) (?:la |las |el )?{_OFFERED}(?:de )?"
    r"\d{1,2} cuotas?\b|"
    rf"\b(?:voy|vamos) con (?:la |las |el )?{_OFFERED}(?:de )?\d{{1,2}} cuotas?\b|"
    r"\bme interesa (?:la )?opcion (?:de )?\d{1,2} cuotas?\b|"
    rf"\bme quedo con (?:la |las |el )?{_OFFERED}(?:de )?\d{{1,2}} cuotas?\b|"
    r"^dale(?:,)? (?:con )?(?:la de )?\d{1,2} cuotas?\b|"
    r"\b(?:la opcion|las?) (?:de )?\d{1,2} cuotas?\b.{0,60}\b(?:me sirve|dejalo asi)\b"
)
_INSTALLMENT_ACCEPTANCE = re.compile(
    r"\b(?:voy a |quiero |podemos )?(?:aceptar|registrar) "
    r"(?:la opcion (?:de )?|las? )?\d{1,2} cuotas?\b"
)
_INSTALLMENT_REJECTION = re.compile(
    r"\b(?:no voy a|no quiero|prefiero no) (?:aceptar|registrar)\b|\bno acepto\b"
)

# Signal lexicons are written per semantic category (§8.2), never as copies of evaluation
# phrasings. `evals/heldout/` measures how far they generalize; the model router only
# classifies what these tables leave as "ambiguo", and it can add an escalation, never remove one.
_DISPUTE = re.compile(
    r"\breclamo\b|\bimpugn|\bdesconozco\b|\bno (?:la )?reconozco\b|"
    r"\b(?:esa|esta|la) deuda no es mia\b|\bno es mi deuda\b|\bnunca (?:saque|contrate|pedi)\b|"
    r"\bno estoy de acuerdo con (?:el|lo que|la)\b|\bdenuncia por el cobro\b|"
    r"\bme (?:estan|estas) cobrando (?:mal|de mas|algo que no)\b|\bcobro indebido\b|"
    r"\b(?:cargo|debito) que (?:yo )?no (?:realice|autorice)\b|"
    r"\bno autorice (?:ningun )?debito\b|\bdebe ser un error de la cuenta\b|"
    r"\bimporte que reclaman no se corresponde\b|\breferencia que me pasan no es mi[oa]\b|"
    r"\bservicio que no figura\b|\bimporte no corresponde (?:a |con )?mi cuenta\b|"
    r"\bmonto (?:esta )?(?:totalmente )?(?:equivocado|incorrecto)\b"
)
_LEGAL = re.compile(
    r"\babogad[oa]s?\b|\bdemand(?:a|ar|arlos|arte)\b|\bjudicial\b|\bjuicio\b|"
    r"\bcarta documento\b|\bdefensa del consumidor\b|\bacciones (?:legales|judiciales)\b|"
    r"\bletrad[oa]\b|\bestudio juridico\b|\bmedidas? cautelares?\b|\bdepartamento legal\b"
)
_CRISIS = re.compile(
    r"\bsuicid|\bquitarme la vida\b|\bmatarme\b|\bno quiero (?:seguir )?viv(?:ir|iendo)\b"
)
_VULNERABILITY = re.compile(
    r"\bvulnerab|"
    # loss of income
    r"\b(?:me quede|quede|estoy|ando) sin (?:trabajo|laburo|empleo|ingresos)\b|"
    r"\bperdi (?:el|mi) (?:trabajo|laburo|empleo)\b|\bme (?:echaron|despidieron)\b|"
    r"\bdesemplead[oa]\b|\bdesocupad[oa]\b|\bno tengo (?:trabajo|laburo|ingresos)\b|"
    r"\bsuspendi(?:do|da|eron) sin sueldo\b|\bno entra (?:un )?peso\b|"
    r"\b(?:me )?rescindieron (?:el|mi) contrato\b|\bme cancelaron (?:el|mi) contrato\b|"
    r"\b(?:me )?(?:bajaron|redujeron|recortaron) (?:las|mis) horas\b|"
    r"\breduccion de (?:horas|jornada)\b|\bcobro (?:la mitad|mucho menos)\b|"
    r"\bchangas? que no alcanzan?\b|"
    # health
    r"\bestoy (?:muy )?enferm[oa]\b|\binternad[oa]\b|\bdiscapacidad\b|\bdepresion\b|"
    r"\b(?:tengo|me diagnosticaron|me detectaron) (?:un )?(?:cancer|tumor)\b|"
    r"\b(?:quimio|quimioterapia|dialisis|tratamiento oncologico)\b|"
    r"\b(?:fractur|lesion|operad)[a-z]*\b.{0,100}\b(?:de baja|no puedo (?:trabajar|laburar))\b|"
    r"\bestoy de baja\b.{0,80}\bno puedo (?:trabajar|laburar|cobrar)\b|"
    r"\baccidente\b.{0,80}\b(?:secuelas|no puedo trabajar|impiden trabajar)\b|"
    r"\bsecuelas\b.{0,80}\b(?:trabajar|laburar)\b|\bprestacion minima\b|"
    # bereavement
    r"\bfallec(?:io|imiento)\b|"
    r"\b(?:se murio|murio|perdi a) (?:mi|mis) (?:mama|papa|madre|padre|hij[oa]s?|espos[oa]|"
    r"pareja|marido|mujer|abuel[oa]|herman[oa])\b|"
    # violence and extreme hardship
    r"\bviolencia\b|\bme (?:pega|golpea|maltrata)\b|"
    r"\bno tengo (?:para comer|que comer)\b|\bme desalojaron\b|\bsituacion de calle\b"
)
_HUMAN = re.compile(
    # "¿puede pagar otra persona por mí?" is FAQ-010, not a transfer request.
    r"\b(?:hablar|atienda|atiende|pasame|pasar|derivame|comunicame) (?:con |a )?(?:una|otra) "
    r"persona\b|\bcon una persona\b|\bun operador\b|\buna operadora\b|\bun[a]? asesor[a]?\b|"
    r"\bun humano\b|\bcon alguien\b|\bpersona real\b|\balguien de carne y hueso\b|"
    r"\balguien de verdad\b|\bquiero (?:a |hablar con )?una persona\b|"
    r"\bpersona que (?:tenga|pueda tener) potestad\b|"
    r"\b(?:manda|pasa|conecta|comunica)me (?:a )?alguien\b|"
    r"\balguien (?:que|con) (?:pueda|potestad para) (?:atender|resolver|revisar)\b|"
    r"\b(?:asigna|asigname|pasa|pasame|dame) (?:un|una|el|la) (?:representante|emplead[oa])\b|"
    r"\b(?:representante|emplead[oa]) que (?:me )?(?:atienda|revise)\b|"
    r"\b(?:telefono|mail) de un[a]? emplead[oa]\b|\brespuestas pregrabadas\b|"
    r"\bno quiero (?:hablar|seguir) con (?:un|el) (?:bot|robot|asistente virtual)\b"
)
# Evidence vocabulary for a model-only "pedido_explicito": who the customer asks for, or the
# automated channel they reject. Categories, not phrasings: the router table above stays the
# deterministic path and this only verifies a quote the classifier already produced.
_HUMAN_ROLE = re.compile(
    r"\b(?:personas?|human[oa]s?|operador(?:a|es|as)?|asesor(?:a|es|as)?|representantes?|"
    r"emplead[oa]s?|supervisor(?:a|es|as)?|encargad[oa]s?|responsables?|gerentes?|alguien)\b"
)
_AUTOMATED_CHANNEL = re.compile(
    r"\b(?:bot|robot|maquina|grabacion(?:es)?|grabad[oa]s?|pregrabad[oa]s?|contestador|"
    r"automatic[oa]s?|asistente virtual|menu(?:es|s)?)\b"
)
# A vulnerability quote must name a grave cause (income, health, loss, violence, basic needs):
# "no llego con esas tres cuotas, lo descarto" rejects a plan, it does not explain a hardship.
_VULNERABILITY_CAUSE = re.compile(
    r"\b(?:trabaj|labur|emple|desemple|desocup|ingres|sueld|salari|changa|despid|echaron|"
    r"suspend|licencia|de baja|horas?\b|rescind|jubil|pension|prestacion|local\b|negocio|"
    r"quiebr|enferm|operac|operar|operaron|internad|hospital|tratamiento|medic|remedio|salud|"
    r"cancer|tumor|quimio|dialisis|accident|fractur|lesion|secuela|discapac|depresi|angusti|"
    r"ansiedad|desesper|no doy mas|psic|fallec|muri|muerte|duelo|luto|viud|violencia|golpe|"
    r"maltrat|desaloj|calle\b|comer\b|comida|alquiler|hij[oa]s?\b|a cargo|embaraz|mango\b)"
)
# A dispute or legal quote must name the dispute or the legal action itself: rejecting a plan
# ("lo rechazo", "no me sirve") is a negotiation answer, not a claim against the debt.
_DISPUTE_EVIDENCE = re.compile(
    r"\bno (?:es|son|era) mi[oa]?s?\b|\bno (?:la |lo )?reconozc|\bdesconozc|\breclam|\bimpugn|"
    r"\b(?:nunca|jamas|no) (?:la |lo )?"
    r"(?:firme|contrate|suscribi|autorice|pedi|use|saque|solicite)\b|"
    r"\b(?:sin|no tiene) mi (?:firma|autorizacion|consentimiento|aprobacion)\b|"
    r"\b(?:cobr|carg)[a-z]* (?:mal|de mas|indebid|por error)|\bindebid|\bfraude|\bestafa|"
    r"\b(?:importe|monto|cargo|cobro|saldo|deuda)\b.{0,20}\b(?:mal|equivocad|incorrect|error)|"
    r"\bno corresponde|\bpor error\b"
)
_LEGAL_EVIDENCE = re.compile(
    r"\babogad|\bletrad|\bdemand|\bjuicio|\bjudicial|\bdenuncia|\bcarta documento\b|"
    r"\bdefensa del consumidor\b|\bjuridic|\blegal"
)
# "¿puede pagar otra persona por mí?" names a person without asking to talk to one (FAQ-010).
_THIRD_PARTY_PAYER = re.compile(
    r"\bpag[a-z]*\b.{0,30}\b(?:otra persona|otro|alguien|familiar|tercero|por mi)\b|"
    r"\b(?:otra persona|alguien|familiar|tercero)\b.{0,30}\bpag"
)
_AMOUNT_AMBIGUITY = re.compile(
    r"\blo que (?:se )?pueda\b|\balgo puedo\b|"
    r"\b(?:no se|ni idea|no tengo idea|no sabria decirte)(?: bien)?(?: de)? cuanto\b|"
    r"\b(?:un poco|algo) (?:por mes|cada mes|todos los meses)\b"
)
_DUE_DATES = re.compile(
    r"venc|\bfechas? de (?:pago|las cuotas)\b|\bcuando (?:tenia|tengo|tendria) que pagar\b"
)
_INTEREST = re.compile(
    r"\binteres(?:es)?\b|\bde que se compone\b|\bcomposicion de (?:la|mi) deuda\b"
)
_OFF_TOPIC = re.compile(
    r"\bmundial\b|\bcapital de\b|\breceta\b|\bfutbol\b|\bclima\b|\bpartido\b|"
    # personal financial advice is out of role (F-02)
    r"\bconsejos? financier|\binvert|\binversion|\bcripto|\bbitcoin\b|\bplazo fijo\b|"
    r"\bcomprar dolares\b|\bacciones\b|\bfondo comun\b|\bfci\b|\bahorros?\b|"
    r"\binflacion\b|\bbonos?\b|\bdepositos?\b|\bmoneda extranjera\b|"
    r"\bcryptos?\b|\bdiversificar\b|"
    # new credit for a purchase is not this debt
    r"\bcompra(?:r)? (?:de )?(?:una |un )?(?:casa|auto|departamento|propiedad|vivienda|terreno)\b|"
    r"\bhipotec"
)
# Instruments that also name an investment. Paying the debt with one asks which payment methods are
# accepted (PAY-MET-003); investing, saving or advice stays out of collections.
_PAYMENT_INSTRUMENT = re.compile(
    r"\b(?:cripto\w*|crypto\w*|bitcoin|ethereum|dolares|moneda extranjera|deposit\w*|cheque)\b"
)
_PAYMENT_ACT = re.compile(r"\b(?:acept|pag|abon|sald|acredit|reflej)\w*")
_INVESTMENT = re.compile(
    r"\binvert|\binversion|\bahorr|\brind|\brendimiento|\bplazo fijo\b|\bcomprar dolares\b|"
    r"\bbonos?\b|\bacciones\b|\bfondo comun\b|\bfci\b|\bdiversificar\b|\bconsejos? financier|"
    r"\binflacion\b"
)


def _has(text: str, *values: str) -> bool:
    return any(value in text for value in values)


def is_amount_ambiguity(text: str) -> bool:
    """Return whether the user wants to pay but has not provided a usable amount."""
    return bool(_AMOUNT_AMBIGUITY.search(detection_skeleton(text)))


def asks_due_dates(text: str) -> bool:
    return bool(_DUE_DATES.search(detection_skeleton(text)))


def asks_debt_composition(text: str) -> bool:
    """FAQ-013: why the balance has interest is answered with the system's composition."""
    normalized = detection_skeleton(text)
    # Interest of a plan (surcharge, rate, discount) is a policy question, not the balance.
    return bool(_INTEREST.search(normalized)) and not _has(
        normalized, "quita", "cuota", "recargo", "tasa", "plan", "perdon", "rebaja"
    )


def declares_crisis(text: str) -> bool:
    return bool(_CRISIS.search(detection_skeleton(text)))


# Vocabulary about paying, the debt or the request itself. A vulnerability quote made only of these
# words says "I can't pay", not why; ESC-002 needs a grave cause to derive.
_PAYMENT_TALK = frozenset(
    "pagar pago pagos pague puedo podemos podria poder puede llego llegar alcanza alcanzan "
    "cubrir total todo toda todos entero completo completa este esta estos mes meses cuota "
    "cuotas opcion opciones alternativa alternativas plan planes quiero quisiera necesito "
    "revisar ver hacer complicado complicada complicados dificil justo justa corto corta plata "
    "dinero deuda deudas saldo monto ahora momento hoy vez estoy ando estamos tengo tenemos hay "
    "con que por porque para una uno unos muy mucho tanto pero mas menos nada solo bien sin "
    # function words and interrogatives never name a cause
    "los las del mis tus sus ese esa eso esto asi tan cuanto cuanta cuantos cuantas cuando como "
    "donde cual quien algo algun alguna todavia tambien".split()
)
_WORDS = re.compile(r"[a-z0-9]+")


def _word_sequence(text: str) -> str:
    return " ".join(_WORDS.findall(detection_skeleton(text)))


# Sentence punctuation and quotation marks, but not the thousands separator of an amount ($5.600).
_QUOTE_BREAK = re.compile(
    r"(?:(?<!\d)[.;!?\u2026]|[.;!?\u2026](?!\d)|[\"\u201c\u201d\u00ab\u00bb])+"
)


def _quoted_fragments(evidence: str) -> list[str]:
    """Clauses of the quote. Models often join or wrap verbatim clauses with periods or quotation
    marks; each clause must still be verbatim, and one-word pieces are not evidence on their own."""
    fragments = [_word_sequence(part) for part in _QUOTE_BREAK.split(evidence)]
    return [fragment for fragment in fragments if fragment]


def escalation_evidence_holds(text: str, evidence: str, signal: str) -> bool:
    """The classifier's quote is in the customer's message and names the signal: a grave cause for
    vulnerability, a person or a rejected automated channel for an explicit request. Guards the
    model-only derivation against over-reading."""
    message = f" {_word_sequence(text)} "
    fragments = _quoted_fragments(evidence)
    if any(f" {fragment} " not in message for fragment in fragments):
        return False
    quote = " ".join(fragment for fragment in fragments if len(fragment.split()) >= 2)
    if len(quote) < 3:
        return False
    if signal == "pedido_explicito":
        if _HUMAN.search(quote):
            return True
        named = _HUMAN_ROLE.search(quote) or _AUTOMATED_CHANNEL.search(quote)
        return bool(named) and not _THIRD_PARTY_PAYER.search(message)
    if signal == "reclamo":
        return bool(_DISPUTE_EVIDENCE.search(quote))
    if signal == "amenaza_legal":
        return bool(_LEGAL_EVIDENCE.search(quote))
    return bool(_VULNERABILITY_CAUSE.search(quote))


_PROPOSAL_REJECT = re.compile(
    r"\bno me (?:sirve|interesa|conviene|cierra)\b|\bmejor no\b|\botras?\b|\bprefiero ver\b|"
    r"^(?:no|nop|nope|nah|paso)\b"
)
_PROPOSAL_ACCEPT = re.compile(
    r"\bme (?:sirve|interesa|conviene|cierra|viene bien|parece bien|la quedo)\b|\besa\b|"
    r"\bla (?:tomo|quiero|acepto)\b|\bde acuerdo\b|\bde una\b|\bpor favor\b|\bveamos\b|"
    r"^(?:si+|sip|dale|ok|oka|okay|okey|okis|perfecto|listo|genial|claro|obvio|bueno|vale|"
    r"porfa|joya|buenisimo|excelente)\b"
)
_PROPOSAL_REPLY_MAX_WORDS = 6
_PLAN_REQUEST = re.compile(
    r"\b(?:quiero|queria|necesito|busco|armar|armame|hacer|haceme|dame|ofreceme|tenes|tienen|hay|"
    r"podemos|puedo tener|me (?:hacen|dan|ofrecen))\b(?:\s+\w+){0,3}\s+plan(?:es)?\b"
)
_CANNOT_PAY = re.compile(
    r"\bno (?:puedo|llego a|voy a poder|me alcanza para) pagar(?:lo)?\b|\bno me alcanza\b|"
    r"\bno llego (?:con|a cubrir)\b|\bno tengo (?:con que|para) pagar\b|"
    r"\bse me (?:hace|complica|dificulta) (?:muy )?(?:dificil|imposible|pagar)\b|"
    r"\bme cuesta pagar\b"
)
_AMOUNT_DIGITS = re.compile(r"(\d{1,3}(?:[.,]\d{3})+|\d+)(?:\s*(mil|k)\b)?")


def monthly_amount(text: str) -> int | None:
    """The amount in a short answer to "¿cuánto podrías pagar?": "1000", "$ 30.000", "unos 30
    mil", "mil pesos". A message about installments is not an amount."""
    normalized = text.casefold()
    if len(normalized.split()) > 8 or "cuota" in normalized:
        return None
    if match := _AMOUNT_DIGITS.search(normalized):
        value = int(re.sub(r"[.,]", "", match.group(1)))
        amount = value * 1000 if match.group(2) else value
    else:
        words = numbers_in_words(detection_skeleton(text))
        amount = int(words[0]) if words else 0
    return amount if amount > 0 else None


_CHOICE_LEAD = r"^(?:(?:elijo|tomo|quiero|prefiero|me quedo con|voy con|vamos con|dame)\s+)?"
_LISTED_INDEX = re.compile(_CHOICE_LEAD + r"(?:la\s+)?(?:opcion\s+)?(?:numero\s+|nro\s+)?(\d)$")
_LISTED_ORDINAL = re.compile(
    _CHOICE_LEAD + r"(?:la\s+)?(primera|segunda|tercera|cuarta|quinta|sexta)(?:\s+opcion)?$"
)
_LISTED_INSTALLMENTS = re.compile(_CHOICE_LEAD + r"la de (\d{1,2})$")
_ORDINALS = {"primera": 1, "segunda": 2, "tercera": 3, "cuarta": 4, "quinta": 5, "sexta": 6}


def listed_choice(text: str) -> tuple[Literal["index", "installments"], int] | None:
    """A choice from the numbered list just shown: "la 4", "la cuarta", "opción 2", "la de 6"."""
    normalized = _word_sequence(text)
    if match := _LISTED_INSTALLMENTS.match(normalized):
        return "installments", int(match.group(1))
    if match := _LISTED_INDEX.match(normalized):
        return "index", int(match.group(1))
    if match := _LISTED_ORDINAL.match(normalized):
        return "index", _ORDINALS[match.group(1)]
    return None


# Rioplatense affirmatives that contain "no" ("no hay problema", "¿por qué no?").
_AFFIRMATIVE_NO = re.compile(r"\bno hay problema\b|\bno pasa nada\b|\bpor que no\b|\bcomo no\b")


_REOPEN_OFFER = re.compile(r"\bsi\b|\bquiero\b|\bmostrame\b|\bveamos\b|\bpasame\b")
_BARE_INSTALLMENTS = re.compile(r"^(?:(?:la|las|el)\s+)?(?:de\s+)?(\d{1,2}) cuotas?$")


def reopens_offer(text: str) -> bool:
    """After "Entendido. Si más adelante querés revisar alternativas…", only a clear yes reopens
    the offer; "bueno" or "ok" there close the exchange."""
    return bool(_REOPEN_OFFER.search(_word_sequence(text)))


def bare_installments(text: str) -> int | None:
    """ "9 cuotas" right after the list is a choice; "¿puedo pagar en 9 cuotas?" is a question."""
    if "?" in text:
        return None
    match = _BARE_INSTALLMENTS.match(_word_sequence(text))
    return int(match.group(1)) if match else None


def proposal_reply(text: str) -> Literal["accept", "reject"] | None:
    """A short answer to "¿Te sirve esa?" about the option just proposed. Longer messages (a
    question about the option, a different request) go through the normal routing table."""
    normalized = " ".join(_WORDS.findall(detection_skeleton(text)))
    if not normalized or len(normalized.split()) > _PROPOSAL_REPLY_MAX_WORDS:
        return None
    if _PROPOSAL_REJECT.search(normalized) or (
        re.search(r"\bno\b", normalized) and not _AFFIRMATIVE_NO.search(normalized)
    ):
        return "reject"
    return "accept" if _PROPOSAL_ACCEPT.search(normalized) else None


# A question about what the policy allows, at the start of the message ("¿Puedo pagar una parte?",
# "¿qué pasa si no pago una cuota?", "ya pagué y me sigue apareciendo"). Asking for concrete
# installments, the options or the balance keeps its own route.
_POLICY_QUESTION = re.compile(
    r"^(?:(?:hola|buenas|buen dia|una consulta|consulta|disculpa)\s+)*(?:y\s+)?(?:"
    r"puedo|me puedo|podria|se puede|se podria|pueden|podrian|me pueden|me podrian|me hacen|"
    r"es posible|aceptan|hay forma de|como hago para|como es|que pasa (?:si|con)|que hacen|"
    r"que recargo|cuantos dias|cuando se (?:actualiza|acredita|refleja)|hasta cuando|"
    r"(?:ya )?(?:pague|hice el pago)|si (?:pago|arranco|firmo|tomo)|tengo que|"
    r"me mandan|me envian)\b"
)
_NOT_A_POLICY_QUESTION = re.compile(
    r"\b(?:cuanto debo|saldo|opcion(?:es)?|alternativas?|en cuotas)\b"
)
_PAY_INTENT = re.compile(
    r"\b(?:quiero|queria|necesito|vengo a|voy a)\s+(?:pagar|abonar|saldar)\b"
    r"(?!\s+(?:el|mi|un|una)\s+(?:plan|acuerdo|cuota))"
)


def asks_policy(text: str, installments: int | None = None) -> bool:
    words = _word_sequence(text)
    return (
        installments is None
        and bool(_POLICY_QUESTION.search(words))
        and not _NOT_A_POLICY_QUESTION.search(words)
        and not _OFF_TOPIC.search(detection_skeleton(text))
    )


def route_turn(text: str) -> RouteResult:
    """Deterministic routing table (§8.2). The model only classifies what this leaves ambiguous."""
    normalized = detection_skeleton(text)

    # Normalize number words only in the installment slot, using the shared Spanish parser.
    def installment_number(match: re.Match[str]) -> str:
        values = numbers_in_words(match.group())
        return (
            f"{int(values[0])} cuotas"
            if len(values) == 1 and 1 <= values[0] <= 99
            else match.group()
        )

    normalized = re.sub(r"\b[a-z]+ cuotas?\b", installment_number, normalized)
    option_match = _OPTION.search(text)
    installments_match = _INSTALLMENTS.search(normalized)
    option_id = option_match.group().upper() if option_match else None
    installments = int(installments_match.group(1)) if installments_match else None

    # ESC-002 before anything else: a declared vulnerability is never negotiated, even when the
    # same message also asks for installments or mentions a dispute.
    if _CRISIS.search(normalized) or _VULNERABILITY.search(normalized):
        return RouteResult(intent="pedido_humano", escalation_motivo="vulnerabilidad")
    if _DISPUTE.search(normalized):
        return RouteResult(intent="pedido_humano", escalation_motivo="reclamo")
    if _LEGAL.search(normalized):
        return RouteResult(intent="pedido_humano", escalation_motivo="amenaza_legal")
    informational_escalation = _has(normalized, "cuando derivan", "motivos de derivacion")
    human_text = (
        re.sub(r"\b(?:un operador|una operadora)\b", "", normalized)
        if informational_escalation
        else normalized
    )
    if _HUMAN.search(human_text):
        return RouteResult(intent="pedido_humano", escalation_motivo="pedido_explicito")
    if _has(normalized, "cuando derivan", "motivos de derivacion"):
        return RouteResult(intent="consulta_general", topic="escalamiento")
    choosing = (
        bool(_INSTALLMENT_CHOICE.search(normalized))
        or bool(_INSTALLMENT_ACCEPTANCE.search(normalized))
        or (
            installments is not None
            and _has(
                normalized,
                "me interesa",
                "me sirve",
                "hacer ese acuerdo",
                "cerrar asi",
                "dejar el acuerdo",
                "dejalo asi",
                "dejarlo asi",
                "podemos dejarlo",
            )
        )
    )
    choosing = choosing and not bool(_INSTALLMENT_REJECTION.search(normalized))
    if option_id or (installments is not None and choosing):
        return RouteResult(intent="aceptar_opcion", option_id=option_id, installments=installments)
    if mixed_request(text):
        return RouteResult(intent="consulta_mixta")
    if _OFF_TOPIC.search(normalized):
        # Paying WITH crypto, a deposit or foreign currency asks which payment methods are accepted.
        # Investing, saving or advice is out of collections even when it mentions installments or
        # "esta deuda".
        if (
            _PAYMENT_INSTRUMENT.search(normalized)
            and _PAYMENT_ACT.search(normalized)
            and not _INVESTMENT.search(normalized)
        ):
            return RouteResult(intent="consulta_general", topic="medios_pago")
        return RouteResult(intent="fuera_de_dominio")
    if policy_request(text):
        topic = "negociacion" if concepts(text) & {"quita", "anticipo"} else "any"
        return RouteResult(intent="consulta_general", topic=topic)
    if _AMOUNT_AMBIGUITY.search(normalized):
        return RouteResult(intent="ambiguo")
    if installments is not None and _has(
        normalized, "cuanto pagaria", "total", "cada cuota", "primera cuota", "primer pago"
    ):
        return RouteResult(intent="negociacion", installments=installments)
    if asks_debt_composition(normalized):
        return RouteResult(intent="consulta_deuda", topic="faq")
    # "interés" as a word: "me interesa" is an answer, not a question about interest. "Perdonar"
    # and "rebaja" name a quita and "adelanto" an anticipo (dev split vocabulary); "perdón" alone
    # is an apology.
    if re.search(r"\binteres(?:es)?\b|\bperdon(?:an|ar|en)\b", normalized) or _has(
        normalized,
        "quita",
        "rebaja",
        "anticipo",
        "adelanto",
        "requisitos para refinanciar",
        "politica de cuotas",
    ):
        return RouteResult(intent="consulta_general", topic="negociacion")
    if asks_policy(text, installments):
        # "¿Puedo pagar una parte de la deuda?" or "¿puedo cambiar la fecha de vencimiento?" ask
        # what the policy allows: the knowledge base answers them, not the balance. No topic is
        # fixed; the section that answers decides the risk.
        return RouteResult(intent="consulta_general", topic="any")
    if _CANNOT_PAY.search(normalized):
        # "no puedo pagar todo" asks for alternatives even without the word "opciones" (N-04).
        return RouteResult(intent="negociacion", installments=installments)
    if _DUE_DATES.search(normalized):
        return RouteResult(intent="consulta_deuda", topic="faq")
    if _has(normalized, "opciones", "alternativas", "cuotas", "negoci"):
        return RouteResult(intent="negociacion", installments=installments)
    if _PLAN_REQUEST.search(normalized):
        # "quiero un plan" asks for alternatives; "plan de ahorro" was already off-topic above and
        # "un cargo por un plan que nunca firmé" names a plan without asking for one.
        return RouteResult(intent="negociacion", installments=installments)
    if re.search(r"\b(?:prescri\w*|sucursal\w*|tasa|cft|costo financiero)\b", normalized):
        return RouteResult(intent="consulta_general")
    if _has(normalized, "cuanto debo", "saldo", "deuda"):
        return RouteResult(intent="consulta_deuda")
    if _has(
        normalized,
        "tarjeta",
        "acredit",
        "transferencia",
        "debito",
        "cupon",
        "medio de pago",
        "medios de pago",
    ):
        return RouteResult(intent="consulta_general", topic="medios_pago")
    if _PAY_INTENT.search(_word_sequence(text)):
        # "hola quiero pagar": the balance and its alternatives, not a payment-method search.
        return RouteResult(intent="consulta_deuda")
    if _has(normalized, "donde llamo", "pagar", "politica"):
        return RouteResult(intent="consulta_general", topic="medios_pago")
    if re.search(r"\b(?:hola|buen dia|buenas|chau|gracias)\b", normalized):
        return RouteResult(intent="saludo_despedida")
    return RouteResult(intent="ambiguo")


# How the customer asks to pay, in a statement: "prefiero pagar por transferencia", "sí, por
# transferencia", "la de 3 cuotas con débito". A question ("¿y si pago con tarjeta?") is answered
# from the knowledge base instead, a method under a negation ("no quiero pagar con débito") is not
# asked for, and "cambia con tarjeta" without a request verb only mentions one.
_METHOD_WORDS: dict[str, MedioPago] = {
    "debito": "debito_automatico",
    "transferencia": "transferencia",
    "tarjeta": "tarjeta",
    "cupon": "cupon",
    "efectivo": "cupon",
}
_METHOD_NAME = r"(debito|transferencia|tarjeta|cupon|efectivo)"
_METHOD_MENTION = re.compile(
    rf"(?:\b(?:pagar|pago|pagarlo|abonar|abono)\s+)?\b(?:por|con|mediante|en|via)\s+"
    rf"(?:el\s+|la\s+)?{_METHOD_NAME}\b|\b(?:prefiero|mejor)\s+(?:el\s+|la\s+)?{_METHOD_NAME}\b"
)
_METHOD_REQUEST_CUE = re.compile(
    r"^(?:si|no|por|con|mejor|prefiero)\b"
    r"|\b(?:quiero|prefiero|mejor|pago|pagar|pagarlo|pagaria|abono|abonar|elijo|tomo|dale|"
    r"cambialo|pasalo|hacelo)\b"
)


def requested_payment_method(text: str) -> MedioPago | None:
    """The one payment method a statement asks for, or None."""
    if "?" in text:
        return None
    words = _word_sequence(text)
    if not _METHOD_REQUEST_CUE.search(words):
        return None
    requested = {
        _METHOD_WORDS[match.group(1) or match.group(2)]
        for match in _METHOD_MENTION.finditer(words)
        if "no" not in words[: match.start()].split()[-2:]
    }
    return next(iter(requested)) if len(requested) == 1 else None
