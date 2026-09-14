from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from app.llm.protocol import LLMClient

# Documented limitation (§11.4, "Lost in Simulation", 2026): success rates move several points
# depending on which model plays the user, and simulated users miss failure patterns that real
# customers trigger. Results are a regression signal, never a replacement for real transcripts.


class Persona(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    customer_id: str = Field(default="CUST-00125", pattern=r"^CUST-\d{5}$")
    persona: str
    objective: str
    success_when: str = "El objetivo declarado se cumplió de forma visible."
    opening: str
    patience: int = Field(ge=1, le=20)
    abandon_if: tuple[str, ...]
    expects_escalation: bool = False
    expects_agreement: bool = False
    success_on_derivation: bool = False
    success_on_option_details: bool = False
    confirmation_doubts_before_yes: int | None = Field(default=None, ge=0, le=5)


class PersonaFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    personas: tuple[Persona, ...]


class SimulatedReply(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_message: str
    outcome: Literal["continue", "success", "abandon"]
    reason: str


class SimulationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user: str
    assistant: str


class SimulationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    persona_id: str
    outcome: Literal["success", "abandon", "patience_exhausted"]
    turns: tuple[SimulationTurn, ...]
    reason: str


def load_personas(path: Path | None = None) -> tuple[Persona, ...]:
    source = path or Path(__file__).with_name("personas.yaml")
    return PersonaFile.model_validate(yaml.safe_load(source.read_text(encoding="utf-8"))).personas


async def simulate_conversation(
    persona: Persona,
    *,
    simulator_llm: LLMClient,
    send_to_agent: Callable[[str], Awaitable[str]],
) -> SimulationResult:
    """Drive a real multi-turn agent through an injected, independently configured user model."""
    transcript: list[SimulationTurn] = []
    user_message = persona.opening
    for _ in range(persona.patience):
        assistant = await send_to_agent(user_message)
        transcript.append(SimulationTurn(user=user_message, assistant=assistant))
        normalized_assistant = assistant.casefold()
        if persona.expects_agreement and "quedó registrado" in normalized_assistant:
            return SimulationResult(
                persona_id=persona.id,
                outcome="success",
                turns=tuple(transcript),
                reason="El agente informó el registro después de la confirmación explícita.",
            )
        if persona.success_on_derivation and "ya te deriv" in normalized_assistant:
            return SimulationResult(
                persona_id=persona.id,
                outcome="success",
                turns=tuple(transcript),
                reason="El agente informó una derivación, que el runner verifica contra las tools.",
            )
        if persona.success_on_option_details and all(
            phrase in normalized_assistant for phrase in ("6 cuotas", "total", "primera cuota")
        ):
            return SimulationResult(
                persona_id=persona.id,
                outcome="success",
                turns=tuple(transcript),
                reason="El agente explicó los términos requeridos de la alternativa.",
            )
        if (
            persona.expects_agreement
            and persona.confirmation_doubts_before_yes is not None
            and "¿confirmás este acuerdo?" in normalized_assistant
        ):
            confirmation_prompts = sum(
                "¿confirmás este acuerdo?" in turn.assistant.casefold() for turn in transcript
            )
            if confirmation_prompts > persona.confirmation_doubts_before_yes:
                # The persona definition, rather than simulator randomness, determines when an
                # explicit yes is finally sent. The following agent turn must still prove that the
                # backend write happened; merely asking for confirmation is never success.
                user_message = "sí, confirmo"
                continue
        history = "\n".join(
            f"Cliente: {turn.user}\nAsistente: {turn.assistant}" for turn in transcript
        )
        decision = await simulator_llm.complete(
            task="simulated_user",
            messages=(
                {
                    "role": "system",
                    "content": (
                        "Actuá como la persona indicada, sin ayudar al asistente ni revelar la "
                        "rúbrica. Avanzá hacia tu objetivo y respondé las preguntas concretas; no "
                        "repitas el mensaje anterior. Marcá success sólo cuando se cumpla "
                        "literalmente la condición de éxito, y abandon sólo si se activa una "
                        "condición de abandono. Si outcome es continue, user_message debe ser "
                        "únicamente el próximo mensaje "
                        "del cliente, sin prefijos Cliente/Asistente ni copiar la transcripción."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Persona: {persona.persona}\nObjetivo: {persona.objective}\n"
                        f"Éxito únicamente cuando: {persona.success_when}\n"
                        f"Abandona si: {list(persona.abandon_if)}\n\nTranscripción:\n{history}"
                    ),
                },
            ),
            response_model=SimulatedReply,
        )
        if decision.outcome == "success" and persona.expects_agreement:
            # The simulated user cannot declare success merely because *it* accepted. The agent
            # must pass through the explicit confirmation gate and report the persisted write.
            if "¿confirmás este acuerdo?" in normalized_assistant:
                user_message = "sí, confirmo"
                continue
            user_message = decision.user_message.strip() or "Quiero la opción de 3 cuotas"
            continue
        if decision.outcome != "continue":
            return SimulationResult(
                persona_id=persona.id,
                outcome=decision.outcome,
                turns=tuple(transcript),
                reason=decision.reason,
            )
        user_message = decision.user_message.strip()
    return SimulationResult(
        persona_id=persona.id,
        outcome="patience_exhausted",
        turns=tuple(transcript),
        reason="Se agotó el presupuesto de turnos de la persona.",
    )
