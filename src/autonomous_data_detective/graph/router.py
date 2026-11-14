
import re
from typing import Literal

from autonomous_data_detective.graph.state import InvestigationState

Route = Literal["profile", "investigate", "both", "clarify"]


def route_question(state: InvestigationState) -> dict:
    """Select an agent path based on the user's question."""
    question = state.get("user_question", "").strip().lower()

    if not question:
        return {"route": "clarify"}

    # Dataset profiling and data-quality intent.
    profile_keywords = (
        "profile",
        "summarize the dataset",
        "summarize dataset",
        "dataset overview",
        "dataset structure",
        "structure of the dataset",
        "overview of the dataset",
        "overview of the data",
        "data quality",
        "missing value",
        "missing values",
        "duplicate row",
        "duplicate rows",
        "duplicate record",
        "duplicate records",
        "duplicates",
        "column details",
        "columns contain",
        "dataset summary",
        "data summary",
    )

    # Analytical intent.
    # Whole-word matching avoids accidental matches such as
    # "sum" inside "summarize".
    investigation_keywords = (
        "calculate",
        "compute",
        "compare",
        "correlation",
        "relationship",
        "highest",
        "lowest",
        "maximum",
        "minimum",
        "largest",
        "smallest",
        "least",
        "greatest",
        "max",
        "min",
        "average",
        "mean",
        "median",
        "middle",
        "mode",
        "standard deviation",
        "variance",
        "total",
        "sum",
        "trend",
        "why",
        "how many",
        "how much",
    )

    def contains_keyword(keyword: str) -> bool:
        pattern = rf"\b{re.escape(keyword)}\b"
        return re.search(pattern, question) is not None

    wants_profile = any(
        contains_keyword(keyword) for keyword in profile_keywords
    )
    wants_investigation = any(
        contains_keyword(keyword) for keyword in investigation_keywords
    )

    if wants_profile and wants_investigation:
        route: Route = "both"
    elif wants_profile:
        route: Route = "profile"
    elif wants_investigation:
        route: Route = "investigate"
    else:
        route: Route = "clarify"

    return {"route": route}
