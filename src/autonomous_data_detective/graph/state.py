from typing import Any, Literal, TypedDict


class InvestigationState(TypedDict, total=False):
    """Shared state for a data investigation workflow."""

    user_question: str
    csv_path: str

    route: Literal["profile", "investigate", "both", "clarify"]

    profile_result: str
    investigation_result: str
    final_response: str

    errors: list[str]
    steps_taken: int

    question_validation: dict[str, Any]
