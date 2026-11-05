from typing import Literal

from autonomous_data_detective.graph.state import InvestigationState

Route = Literal["profile", "investigate", "both", "clarify"]


def route_question(state: InvestigationState) -> dict:
    """Select an agent path based on the user's question."""
    question = state.get("user_question", "").strip().lower()

    if not question:
        return {"route": "clarify"}

    profile_keywords = (
        "profile",
        "dataset overview",
        "dataset structure",
        "missing values",
        "duplicate rows",
        "data quality",
        "column details",
    )

    investigation_keywords = (
        "calculate",
        "compare",
        "correlation",
        "relationship",
        "highest",
        "lowest",
        "average",
        "total",
        "sum",
        "trend",
        "why",
        "which",
        "how many",
    )

    wants_profile = any(
        keyword in question for keyword in profile_keywords
    )
    wants_investigation = any(
        keyword in question for keyword in investigation_keywords
    )

    if wants_profile and wants_investigation:
        route: Route = "both"
    elif wants_profile:
        route = "profile"
    elif wants_investigation:
        route = "investigate"
    else:
        route = "clarify"

    return {"route": route}
