from types import SimpleNamespace

import pytest

from autonomous_data_detective.graph.graph import finalize_node
from autonomous_data_detective.graph.nodes import (
    investigate_node,
    profile_node,
)


class FakeAgent:
    """Simulate an agent that fails when invoked."""

    def invoke(self, inputs):
        raise RuntimeError("Simulated agent failure")


def test_profiler_failure_is_recorded(monkeypatch):
    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_profiler_agent",
        lambda: FakeAgent(),
    )

    state = {
        "csv_path": "data/samples/sales.csv",
        "steps_taken": 0,
    }

    result = profile_node(state)

    assert "profile_result" not in result
    assert result["steps_taken"] == 1
    assert len(result["errors"]) == 1
    assert "Profiler failed" in result["errors"][0]
    assert "Simulated agent failure" in result["errors"][0]


def test_investigator_failure_preserves_existing_errors(monkeypatch):
    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        lambda: FakeAgent(),
    )

    state = {
        "user_question": "Which product has the highest revenue?",
        "csv_path": "data/samples/sales.csv",
        "steps_taken": 1,
        "errors": ["Previous warning"],
    }

    result = investigate_node(state)

    assert result["steps_taken"] == 2
    assert len(result["errors"]) == 2
    assert result["errors"][0] == "Previous warning"
    assert "Investigator failed" in result["errors"][1]


def test_finalize_preserves_successful_results_when_error_exists():
    state = {
        "profile_result": "10 rows and 5 columns.",
        "errors": ["Investigator failed: RuntimeError"],
    }

    result = finalize_node(state)

    assert "10 rows and 5 columns." in result["final_response"]
    assert "Workflow Warnings" in result["final_response"]
    assert "Investigator failed" in result["final_response"]


def test_step_limit_error_is_not_swallowed(monkeypatch):
    class FakeSettings:
        max_investigation_steps = 2

    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.get_settings",
        lambda: FakeSettings(),
    )

    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_profiler_agent",
        lambda: FakeAgent(),
    )

    state = {
        "csv_path": "data/samples/sales.csv",
        "steps_taken": 2,
    }

    with pytest.raises(
        RuntimeError,
        match="Maximum investigation steps reached: 2",
    ):
        profile_node(state)


def test_investigator_receives_profiler_findings(monkeypatch):
    captured_inputs = {}

    class FakeAgent:
        def invoke(self, inputs):
            captured_inputs.update(inputs)
            return {
                "messages": [
                    type(
                        "Message",
                        (),
                        {"content": "Revenue analysis completed."},
                    )()
                ]
            }

    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        lambda: FakeAgent(),
    )

    state = {
        "user_question": "Compare revenue by region.",
        "csv_path": "data/samples/sales.csv",
        "profile_result": (
            "The dataset has 10 rows. "
            "The revenue column contains missing values."
        ),
        "steps_taken": 0,
    }

    from autonomous_data_detective.graph.nodes import investigate_node

    result = investigate_node(state)

    user_message = captured_inputs["messages"][0]["content"]

    assert "The dataset has 10 rows." in user_message
    assert "The revenue column contains missing values." in user_message
    assert result["investigation_result"] == "Revenue analysis completed."
    assert result["steps_taken"] == 1



def test_investigator_works_without_profiler_findings(monkeypatch):
    captured_inputs = {}

    class FakeAgent:
        def invoke(self, inputs):
            captured_inputs.update(inputs)
            return {
                "messages": [
                    type(
                        "Message",
                        (),
                        {"content": "Revenue analysis completed."},
                    )()
                ]
            }

    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        lambda: FakeAgent(),
    )

    state = {
        "user_question": "Compare revenue by region.",
        "csv_path": "data/samples/sales.csv",
        "steps_taken": 0,
    }

    from autonomous_data_detective.graph.nodes import investigate_node

    result = investigate_node(state)

    user_message = captured_inputs["messages"][0]["content"]

    assert "No profiler findings are available." in user_message
    assert "Use the available analysis tools" in user_message
    assert result["investigation_result"] == "Revenue analysis completed."
    assert result["steps_taken"] == 1


def test_profiler_construction_failure_is_recorded(monkeypatch):
    def fail_to_build_agent():
        raise RuntimeError("Simulated agent construction failure")

    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_profiler_agent",
        fail_to_build_agent,
    )

    state = {
        "csv_path": "data/samples/sales.csv",
        "steps_taken": 0,
    }

    result = profile_node(state)

    assert "profile_result" not in result
    assert result["steps_taken"] == 1
    assert len(result["errors"]) == 1
    assert "Profiler failed" in result["errors"][0]
    assert "Simulated agent construction failure" in result["errors"][0]


def test_profiler_converts_non_string_result_to_string(monkeypatch):
    class FakeAgent:
        def invoke(self, inputs):
            return {
                "messages": [
                    SimpleNamespace(content={"rows": 10, "columns": 5})
                ]
            }

    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_profiler_agent",
        lambda: FakeAgent(),
    )

    result = profile_node(
        {
            "csv_path": "data/samples/sales.csv",
            "steps_taken": 0,
        }
    )

    assert result["profile_result"] == "{'rows': 10, 'columns': 5}"
    assert result["steps_taken"] == 1


def test_investigator_converts_unexpected_structured_response_to_string(
    monkeypatch,
):
    class FakeAgent:
        def invoke(self, inputs):
            return {
                "structured_response": {"answer": "Revenue is highest in North"},
                "messages": [],
            }

    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        lambda: FakeAgent(),
    )

    result = investigate_node(
        {
            "user_question": "Which region has the highest revenue?",
            "csv_path": "data/samples/sales.csv",
            "steps_taken": 0,
            "errors": [],
        }
    )

    assert result["investigation_result"] == (
        "{'answer': 'Revenue is highest in North'}"
    )
    assert result["steps_taken"] == 1


def test_investigator_converts_non_string_message_content_to_string(
    monkeypatch,
):
    class FakeAgent:
        def invoke(self, inputs):
            return {
                "messages": [
                    SimpleNamespace(content={"answer": 42})
                ]
            }

    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        lambda: FakeAgent(),
    )

    result = investigate_node(
        {
            "user_question": "What is the answer?",
            "csv_path": "data/samples/sales.csv",
            "steps_taken": 0,
            "errors": [],
        }
    )

    assert result["investigation_result"] == "{'answer': 42}"
    assert result["steps_taken"] == 1
