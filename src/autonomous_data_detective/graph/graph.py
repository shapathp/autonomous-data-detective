
from langgraph.graph import END, START, StateGraph

from autonomous_data_detective.graph.nodes import (
    investigate_node,
    profile_node,
)
from autonomous_data_detective.graph.question_validation import (
    validate_question,
)
from autonomous_data_detective.graph.router import route_question
from autonomous_data_detective.graph.state import InvestigationState


def finalize_node(state: InvestigationState) -> dict:
    """Combine successful results and report any errors."""
    sections = []

    profile_result = state.get("profile_result")
    investigation_result = state.get("investigation_result")
    errors = state.get("errors", [])

    if profile_result:
        sections.append(f"## Dataset Profile\n\n{profile_result}")

    if investigation_result:
        sections.append(
            f"## Investigation Findings\n\n{investigation_result}"
        )

    if errors:
        error_details = "\n".join(f"- {error}" for error in errors)

        sections.append(
            "## Workflow Warnings\n\n"
            "Some parts of the investigation could not be completed:\n\n"
            f"{error_details}"
        )

    final_response = "\n\n".join(sections)

    if not final_response:
        final_response = (
            "The workflow completed without producing results "
            "or recording a specific error."
        )

    return {"final_response": final_response}




def clarify_node(state: InvestigationState) -> dict:
    """Provide a targeted clarification when possible."""
    user_question = state.get("user_question")
    csv_path = state.get("csv_path")

    if user_question and csv_path:
        validation = validate_question(user_question, csv_path)

        if validation.get("status") in {"clarify", "invalid"}:
            message = validation.get("message")
            if message:
                return {"final_response": message}

    return {
        "final_response": (
            "I can help investigate your CSV dataset. "
            "Please specify what you want to measure or compare. "
            "For example, ask 'Which product has the highest revenue?' "
            "or 'Show missing values and duplicate rows.' "
            "You can also ask which group has the highest average "
            "of a numeric column."
        )
    }


def question_validator_node(state: InvestigationState) -> dict:
    """Validate an investigation question against the CSV schema."""
    result = validate_question(
        state["user_question"],
        state["csv_path"],
    )

    return {
        "question_validation": result,
    }


def validation_response_node(state: InvestigationState) -> dict:
    """Return an explanatory response when validation fails."""
    validation = state.get("question_validation", {})

    message = validation.get(
        "message",
        "The question could not be validated.",
    )

    return {"final_response": message}


def after_question_validation(state: InvestigationState) -> str:
    """Choose the next node based on validation and original routing."""
    validation = state.get("question_validation", {})

    # Invalid, ambiguous, or failed validation must not reach the investigator.
    if validation.get("status") != "valid":
        return "validation_response"

    # A request needing both profiling and investigation starts with profiling.
    if state["route"] == "both":
        return "profiler"

    return "investigator"


def after_profiling(state: InvestigationState) -> str:
    """Investigate further only when the original route requests both."""
    if state["route"] == "both":
        return "investigator"

    return "finalize"


def build_investigation_graph():
    """Build and compile the data investigation workflow."""
    builder = StateGraph(InvestigationState)

    # Register workflow nodes.
    builder.add_node("router", route_question)
    builder.add_node("profiler", profile_node)
    builder.add_node("investigator", investigate_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("question_validator", question_validator_node)
    builder.add_node("validation_response", validation_response_node)
    builder.add_node("finalize", finalize_node)

    # Every request starts with routing.
    builder.add_edge(START, "router")

    # Profile-only requests keep the existing direct profiling path.
    # Investigation requests must pass dataset-aware validation first.
    builder.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "profile": "profiler",
            "investigate": "question_validator",
            "both": "question_validator",
            "clarify": "clarify",
        },
    )

    # Decide whether a validated question can proceed.
    builder.add_conditional_edges(
        "question_validator",
        after_question_validation,
        {
            "profiler": "profiler",
            "investigator": "investigator",
            "validation_response": "validation_response",
        },
    )

    # After profiling, continue to investigation only for the "both" route.
    builder.add_conditional_edges(
        "profiler",
        after_profiling,
        {
            "investigator": "investigator",
            "finalize": "finalize",
        },
    )

    # Complete each workflow branch.
    builder.add_edge("investigator", "finalize")
    builder.add_edge("clarify", END)
    builder.add_edge("validation_response", END)
    builder.add_edge("finalize", END)

    return builder.compile()
