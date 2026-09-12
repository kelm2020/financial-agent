from enum import StrEnum

from pydantic import BaseModel

from app.tools.schemas import MODEL_TOOL_SCHEMAS


class ToolPhase(StrEnum):
    NORMAL = "normal"
    DRAFT_PENDING = "draft_pending"
    DRAFT_QUESTION = "draft_question"


_NORMAL_TOOLS = frozenset(MODEL_TOOL_SCHEMAS)
_DRAFT_QUESTION_TOOLS = frozenset({"get_debt", "get_payment_options", "search_policies"})


def model_tools_for_phase(phase: ToolPhase) -> dict[str, type[BaseModel]]:
    """Return only capabilities that may be exposed to the model in this phase."""
    if phase is ToolPhase.DRAFT_PENDING:
        return {}
    names = _DRAFT_QUESTION_TOOLS if phase is ToolPhase.DRAFT_QUESTION else _NORMAL_TOOLS
    return {name: MODEL_TOOL_SCHEMAS[name] for name in names}
