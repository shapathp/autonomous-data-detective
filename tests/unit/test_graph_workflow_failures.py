from autonomous_data_detective.graph.graph import (
    build_investigation_graph,
)


class FakeAgent:
    """Simulate an agent that fails during execution."""

    def invoke(self, inputs):
        raise RuntimeError("Simulated OpenAI failure")


def test_graph_preserves_profile_when_investigator_fails(monkeypatch):
    # The profiler succeeds, but the investigator fails.
    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_profiler_agent",
        lambda: type(
            "SuccessfulAgent",
            (),
            {
                "invoke": lambda self, inputs: {
                    "messages": [
                        type(
                            "Message",
                            (),
                            {"content": "Dataset has 10 rows and 5 columns."},
                        )()
                    ]
                }
            },
        )(),
    )

    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        lambda: FakeAgent(),
    )

    graph = build_investigation_graph()

    result = graph.invoke(
        {
            "user_question": (
                "Profile the sales dataset and identify "
                "the product with the highest total revenue."
            ),
            "csv_path": "data/samples/sales.csv",
            "steps_taken": 0,
        }
    )

    assert result["route"] == "both"
    assert "10 rows and 5 columns" in result["profile_result"]
    assert "investigation_result" not in result
    assert any(
        "Investigator failed" in error
        for error in result["errors"]
    )
    assert "Dataset Profile" in result["final_response"]
    assert "Workflow Warnings" in result["final_response"]


def test_graph_finalizes_when_profiler_fails(monkeypatch):
    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_profiler_agent",
        lambda: FakeAgent(),
    )

    graph = build_investigation_graph()

    result = graph.invoke(
        {
            "user_question": "Give me a profile of the dataset.",
            "csv_path": "data/samples/sales.csv",
            "steps_taken": 0,
        }
    )

    assert result["route"] == "profile"
    assert "profile_result" not in result
    assert any(
        "Profiler failed" in error
        for error in result["errors"]
    )
    assert "Workflow Warnings" in result["final_response"]

def test_graph_records_both_failures(monkeypatch):
    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_profiler_agent",
        lambda: FakeAgent(),
    )
    monkeypatch.setattr(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        lambda: FakeAgent(),
    )

    graph = build_investigation_graph()

    result = graph.invoke(
        {
            "user_question": (
                "Profile the sales dataset and identify "
                "the product with the highest total revenue."
            ),
            "csv_path": "data/samples/sales.csv",
            "steps_taken": 0,
        }
    )

    assert result["route"] == "both"
    assert "profile_result" not in result
    assert "investigation_result" not in result

    errors = result["errors"]
    assert len(errors) == 2
    assert any("Profiler failed" in error for error in errors)
    assert any("Investigator failed" in error for error in errors)

    assert "Workflow Warnings" in result["final_response"]
    assert "Profiler failed" in result["final_response"]
    assert "Investigator failed" in result["final_response"]