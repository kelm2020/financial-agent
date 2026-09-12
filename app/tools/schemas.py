from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GetCustomerArgs(BaseModel):
    """Obtiene el perfil del cliente autenticado de esta conversación."""

    model_config = ConfigDict(extra="forbid")

    incluir_contacto: bool


class GetDebtArgs(BaseModel):
    """Consulta el detalle de deuda del cliente autenticado."""

    model_config = ConfigDict(extra="forbid")

    incluir_historial: bool


class GetPaymentOptionsArgs(BaseModel):
    """Consulta opciones vigentes; no crea ni modifica acuerdos."""

    model_config = ConfigDict(extra="forbid")

    incluir_detalle: bool


class SearchPoliciesArgs(BaseModel):
    """Busca reglas generales; nunca datos de una cuenta."""

    model_config = ConfigDict(extra="forbid")

    query: str
    topic: Literal["negociacion", "medios_pago", "escalamiento", "faq", "any"]


class ProposeAgreementArgs(BaseModel):
    """Propone una opción ya ofrecida, sin registrar ningún acuerdo."""

    model_config = ConfigDict(extra="forbid")

    opcion_id: str


class RequestHumanArgs(BaseModel):
    """Solicita derivación a un operador humano."""

    model_config = ConfigDict(extra="forbid")

    motivo: Literal[
        "pedido_explicito",
        "reclamo",
        "fuera_de_politica",
        "vulnerabilidad",
        "falla_tecnica",
        "loop_sin_avance",
        "identidad_no_verificada",
        "amenaza_legal",
        "outcome_de_escritura_desconocido",
    ]


MODEL_TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "get_customer": GetCustomerArgs,
    "get_debt": GetDebtArgs,
    "get_payment_options": GetPaymentOptionsArgs,
    "search_policies": SearchPoliciesArgs,
    "propose_agreement": ProposeAgreementArgs,
    "request_human": RequestHumanArgs,
}


class StrictDomainModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class PreviousAgreements(StrictDomainModel):
    total: int = Field(ge=0)
    incumplidos: int = Field(ge=0)


class Customer(StrictDomainModel):
    customer_id: str = Field(pattern=r"^CUST-\d{5}$")
    nombre: str
    apellido: str
    idioma: Literal["es-AR"]
    identidad_verificada: bool
    email_registrado: bool
    marca_vulnerabilidad: bool
    acuerdos_activos: int = Field(ge=0)
    acuerdos_previos: PreviousAgreements
    en_gestion_judicial: bool = False


class DueItem(StrictDomainModel):
    periodo: str = Field(pattern=r"^\d{4}-\d{2}$")
    monto: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    vencimiento: date
    estado: Literal["vencido", "pendiente", "pagado"]


class LastPayment(StrictDomainModel):
    fecha: date
    monto: Decimal = Field(gt=0, max_digits=12, decimal_places=2)


class Debt(StrictDomainModel):
    customer_id: str = Field(pattern=r"^CUST-\d{5}$")
    moneda: Literal["ARS"]
    saldo_total: Decimal = Field(ge=0, max_digits=12, decimal_places=2)
    capital: Decimal = Field(ge=0, max_digits=12, decimal_places=2)
    intereses: Decimal = Field(ge=0, max_digits=12, decimal_places=2)
    dias_mora: int = Field(ge=0)
    estado: Literal["mora_temprana", "mora_media", "mora_tardia", "prejudicial", "paid"]
    vencimientos: list[DueItem]
    acuerdos_previos: PreviousAgreements | None = None
    ultimo_pago: LastPayment | None = None
    as_of: datetime


class PaymentOption(StrictDomainModel):
    opcion_id: str = Field(pattern=r"^OPT-[A-Z0-9]{2,10}$")
    tipo: Literal["pago_unico", "cuotas"]
    cuotas: int = Field(ge=1, le=24)
    anticipo: Decimal = Field(ge=0, max_digits=12, decimal_places=2)
    quita_interes: Decimal = Field(ge=0, max_digits=12, decimal_places=2)
    recargo_pct: Decimal = Field(ge=0, le=100)
    monto_financiado: Decimal | None = Field(default=None, ge=0)
    monto_cuota: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    monto_total: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    primer_vencimiento: date
    valid_until: datetime
    policy_refs: list[str]


class PaymentOptions(StrictDomainModel):
    customer_id: str = Field(pattern=r"^CUST-\d{5}$")
    opciones: list[PaymentOption]
    motivo: str | None = None
    policy_refs: list[str] = Field(default_factory=list)
    as_of: datetime


class AgreementDraft(StrictDomainModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    draft_id: str
    opcion_id: str = Field(pattern=r"^OPT-[A-Z0-9]{2,10}$")
    monto_total: Decimal = Field(gt=0)
    cuotas: int = Field(ge=1, le=24)
    monto_cuota: Decimal = Field(gt=0)
    fecha_primer_vencimiento: date
    medio_pago: Literal["debito_automatico", "transferencia", "tarjeta", "cupon"]
    debt_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    expires_at: datetime
    policy_refs: list[str]


class OptionsSnapshot(StrictDomainModel):
    options: list[PaymentOption]
    fetched_at: datetime
    debt_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class CreateAgreementRequest(StrictDomainModel):
    customer_id: str = Field(pattern=r"^CUST-\d{5}$")
    draft_id: str
    opcion_id: str = Field(pattern=r"^OPT-[A-Z0-9]{2,10}$")
    debt_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    medio_pago: Literal["debito_automatico", "transferencia", "tarjeta", "cupon"]


class AgreementResponse(StrictDomainModel):
    agreement_id: str
    customer_id: str
    draft_id: str
    opcion_id: str
    debt_fingerprint: str
    estado: Literal["active"]
    created_at: datetime
    replayed: bool = False


class TransferRequest(StrictDomainModel):
    customer_id: str = Field(pattern=r"^CUST-\d{5}$")
    conversation_id: str
    motivo: str
    resumen: str


class TransferResponse(StrictDomainModel):
    ticket_id: str
    customer_id: str
    estado: Literal["queued"]
    created_at: datetime


ToolStatus = Literal[
    "ok",
    "not_found",
    "invalid_input",
    "partial",
    "upstream_error",
    "timeout",
    "rejected_by_policy",
]


class ToolResult[T](BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ToolStatus
    data: T | None = None
    partial_data: dict[str, Any] | None = None
    message_for_model: str
    retriable: bool = False
    correlation_id: str
