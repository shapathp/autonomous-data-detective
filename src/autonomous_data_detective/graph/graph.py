from langgraph.graph import END, START, StateGraph

from autonomous_data_detective.graph.nodes import (
    investigate_node,
    profile_node,
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
        sections.append(
            f"## Dataset Profile\n\n{profile_result}"
        )

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
    """Ask the user to provide a supported data-analysis question."""
    return {
        "final_response": (
            "I can help investigate your CSV dataset. "
            "Please ask a question such as 'Which product has "
            "the highest revenue?' or 'Show missing values "
            "and duplicate rows.'"
        )
    }

def build_investigation_graph():
    """Build and compile the data investigation workflow."""

    builder = StateGraph(InvestigationState)

    builder.add_node("router", route_question)
    builder.add_node("profiler", profile_node)
    builder.add_node("investigator", investigate_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "profile": "profiler",
            "investigate": "investigator",
            "both": "profiler",
            "clarify": "clarify"
        },
    )

    # After profiling, run the Investigator only for the "both" route.
    builder.add_conditional_edges(
        "profiler",
        lambda state: (
            "investigator"
            if state["route"] == "both"
            else "finalize"
        ),
        {
            "investigator": "investigator",
            "finalize": "finalize",
        },
    )

    builder.add_edge("investigator", "finalize")
    builder.add_edge("clarify", END)
    builder.add_edge("finalize", END)

    return builder.compile()
