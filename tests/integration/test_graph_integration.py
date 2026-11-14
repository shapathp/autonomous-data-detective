
from types import SimpleNamespace

from autonomous_data_detective.graph.graph import (
    build_investigation_graph,
)


class FakeAgent:
    """Return predictable results without making API calls."""

    def __init__(self, response):
        self.response = response

    def invoke(self, inputs):
        return {
            "messages": [
                SimpleNamespace(content=self.response)
            ]
        }


def test_profile_route_completes_successfully(monkeypatch):
    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_profiler_agent",
        lambda: FakeAgent("Dataset has 10 rows and 5 columns."),
    )

    graph = build_investigation_graph()

    result = graph.invoke({
        "user_question": "Give me a profile of the dataset.",
        "csv_path": "data/samples/sales.csv",
        "steps_taken": 0,
    })

    assert result["route"] == "profile"
    assert "10 rows and 5 columns" in result["profile_result"]
    assert "Dataset Profile" in result["final_response"]
    assert "Workflow Warnings" not in result["final_response"]



def test_investigate_route_completes_successfully(monkeypatch):
    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        lambda: FakeAgent("Laptop has the highest total revenue."),
    )

    monkeypatch.setattr(
        "autonomous_data_detective.graph.graph.validate_question",
        lambda question, csv_path: {
            "status": "valid",
            "message": "The question passed dataset schema validation.",
        },
    )

    graph = build_investigation_graph()

    result = graph.invoke({
        "user_question": "Which product has the highest revenue?",
        "csv_path": "data/samples/sales.csv",
        "steps_taken": 0,
    })

    assert result["route"] == "investigate"
    assert (
        result["investigation_result"]
        == "Laptop has the highest total revenue."
    )
    assert result["final_response"]


def test_both_route_completes_successfully(monkeypatch):
    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_profiler_agent",
        lambda: FakeAgent("Dataset has 10 rows and 5 columns."),
    )
    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        lambda: FakeAgent("North has the highest average revenue."),
    )

    graph = build_investigation_graph()

    result = graph.invoke({
        "user_question": (
            "Profile the dataset and compare revenue by region."
        ),
        "csv_path": "data/samples/sales.csv",
        "steps_taken": 0,
    })

    assert result["route"] == "both"
    assert "10 rows and 5 columns" in result["profile_result"]
    assert (
        result["investigation_result"]
        == "North has the highest average revenue."
    )

    response = result["final_response"]
    assert "Dataset Profile" in response
    assert "Investigation Findings" in response
    assert "Workflow Warnings" not in response


def test_clarify_route_completes_successfully():
    graph = build_investigation_graph()

    result = graph.invoke({
        "user_question": "Hello!",
        "csv_path": "data/samples/sales.csv",
        "steps_taken": 0,
    })

    assert result["route"] == "clarify"
    assert "CSV dataset" in result["final_response"]
    assert "highest revenue" in result["final_response"]


def test_validation_error_stops_investigation(monkeypatch):
    monkeypatch.setattr(
        "autonomous_data_detective.graph.graph.validate_question",
        lambda question, csv_path: {
            "status": "error",
            "error_type": "RuntimeError",
            "message": "Question validation failed because the API is unavailable.",
        },
    )

    def unexpected_investigator(*args, **kwargs):
        raise AssertionError("Investigator should not run after validation fails.")

    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        unexpected_investigator,
    )

    graph = build_investigation_graph()

    result = graph.invoke({
        "user_question": "Which product has the highest revenue?",
        "csv_path": "data/samples/sales.csv",
        "steps_taken": 0,
    })

    assert result["route"] == "investigate"
    assert result["final_response"] == (
        "Question validation failed because the API is unavailable."
    )
    assert "investigation_result" not in result