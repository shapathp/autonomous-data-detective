
from autonomous_data_detective.graph.graph import (
    clarify_node,
    finalize_node,
)


def test_finalize_node_combines_profile_and_investigation_results():
    result = finalize_node(
        {
            "profile_result": "Dataset contains 10 rows.",
            "investigation_result": "North has the highest revenue.",
        }
    )

    assert "## Dataset Profile" in result["final_response"]
    assert "Dataset contains 10 rows." in result["final_response"]
    assert "## Investigation Findings" in result["final_response"]
    assert "North has the highest revenue." in result["final_response"]


def test_finalize_node_includes_multiple_workflow_errors():
    result = finalize_node(
        {
            "errors": [
                "Profiler failed.",
                "Investigator failed.",
            ]
        }
    )

    response = result["final_response"]

    assert "## Workflow Warnings" in response
    assert "- Profiler failed." in response
    assert "- Investigator failed." in response


def test_finalize_node_returns_fallback_when_state_is_empty():
    result = finalize_node({})

    assert result["final_response"] == (
        "The workflow completed without producing results "
        "or recording a specific error."
    )


def test_clarify_node_returns_supported_question_examples():
    result = clarify_node({})

    response = result["final_response"]

    assert "CSV dataset" in response
    assert "highest revenue" in response
    assert "missing values" in response
