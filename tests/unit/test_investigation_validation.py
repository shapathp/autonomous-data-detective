import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from autonomous_data_detective.agents.schemas import (
    FactualClaim,
    InvestigationReport,
)
from autonomous_data_detective.graph.nodes import investigate_node


def make_report(claims):
    """Create a structured report for testing."""
    return InvestigationReport(
        summary="North had the highest average revenue.",
        findings=["North had the highest average revenue."],
        evidence=["Group comparison tool output."],
        limitations=[],
        recommendations=[],
        claims=claims,
    )


def make_tool_message(output):
    """Create a simulated successful tool response."""
    return ToolMessage(
        content=json.dumps(output),
        tool_call_id="test_tool_call",
        name="compare_groups_tool",
    )


def test_investigate_node_validates_supported_group_comparison():
    claim = FactualClaim(
        claim_id="claim_1",
        statement="North had the highest average revenue.",
        claim_type="group_comparison",
        claim_data={
            "group": "North",
            "value": 56000,
            "group_column": "region",
            "metric_column": "revenue",
            "aggregation": "mean",
        },
        evidence_ids=["model_supplied_id"],
    )

    report = make_report([claim])

    tool_message = make_tool_message(
        {
            "group_by": "region",
            "metric": "revenue",
            "aggregation": "mean",
            "results": [
                {"group": "North", "value": 56000},
                {"group": "West", "value": 50000},
            ],
        }
    )

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": report,
        "messages": [tool_message],
    }

    with patch(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        return_value=fake_agent,
    ):
        result = investigate_node(
            {
                "user_question": "Which region has the highest average revenue?",
                "csv_path": "sales.csv",
                "profile_result": "",
                "steps_taken": 0,
                "errors": [],
            }
        )

    output = json.loads(result["investigation_result"])

    assert len(output["validation_results"]) == 1
    validation = output["validation_results"][0]

    assert validation["claim_id"] == "claim_1"
    assert validation["status"] == "verified"
    assert validation["evidence_ids"] == ["evidence_1"]


def test_investigate_node_marks_claim_unverifiable_without_matching_evidence():
    claim = FactualClaim(
        claim_id="claim_2",
        statement="North had the highest average revenue.",
        claim_type="group_comparison",
        claim_data={
            "group": "North",
            "value": 56000,
            "group_column": "region",
            "metric_column": "revenue",
            "aggregation": "mean",
        },
        evidence_ids=["model_supplied_id"],
    )

    report = make_report([claim])

    # This tool output refers to a different metric, so it should not match.
    tool_message = make_tool_message(
        {
            "group_by": "region",
            "metric": "quantity",
            "aggregation": "mean",
            "results": [
                {"group": "North", "value": 56000},
            ],
        }
    )

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": report,
        "messages": [tool_message],
    }

    with patch(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        return_value=fake_agent,
    ):
        result = investigate_node(
            {
                "user_question": "Which region has the highest average revenue?",
                "csv_path": "sales.csv",
                "profile_result": "",
                "steps_taken": 0,
                "errors": [],
            }
        )

    output = json.loads(result["investigation_result"])

    assert len(output["validation_results"]) == 1
    validation = output["validation_results"][0]

    assert validation["claim_id"] == "claim_2"
    assert validation["status"] == "unverifiable"
    assert validation["evidence_ids"] == []



def test_investigate_node_marks_incorrect_value_unverifiable():
    claim = FactualClaim(
        claim_id="claim_3",
        statement="North had the highest average revenue of 70000.",
        claim_type="group_comparison",
        claim_data={
            "group": "North",
            "value": 70000,
            "group_column": "region",
            "metric_column": "revenue",
            "aggregation": "mean",
        },
        evidence_ids=["model_supplied_id"],
    )

    report = make_report([claim])

    tool_message = make_tool_message(
        {
            "group_by": "region",
            "metric": "revenue",
            "aggregation": "mean",
            "results": [
                {"group": "North", "value": 56000},
                {"group": "West", "value": 50000},
            ],
        }
    )

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": report,
        "messages": [tool_message],
    }

    with patch(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        return_value=fake_agent,
    ):
        result = investigate_node(
            {
                "user_question": "Which region has the highest average revenue?",
                "csv_path": "sales.csv",
                "profile_result": "",
                "steps_taken": 0,
                "errors": [],
            }
        )

    output = json.loads(result["investigation_result"])
    validation = output["validation_results"][0]

    assert validation["claim_id"] == "claim_3"
    assert validation["status"] == "unverifiable"


def test_numerical_claim_ignores_unrelated_tool_evidence():
    from autonomous_data_detective.agents.schemas import EvidenceRecord
    from autonomous_data_detective.graph.nodes import find_evidence_for_claim

    claim = FactualClaim(
        claim_id="claim_4",
        statement="Revenue mean is 38555.56.",
        claim_type="numerical_result",
        claim_data={
            "column_name": "revenue",
            "result_key": "mean",
            "value": 38555.56,
        },
        evidence_ids=["evidence_1"],
    )

    unrelated_evidence = EvidenceRecord(
        evidence_id="evidence_1",
        tool_name="analyze_missing_values_tool",
        tool_output={
            "column_name": "revenue",
            "mean": 38555.56,
            "statistics": {"mean": 38555.56},
        },
    )

    matching_evidence = find_evidence_for_claim(
        claim,
        [unrelated_evidence],
    )

    assert matching_evidence == []



def test_investigate_node_validates_correlation_claim():
    claim = FactualClaim(
        claim_id="claim_correlation_1",
        statement="Revenue and quantity have a strong positive correlation.",
        claim_type="correlation",
        claim_data={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
            "correlation": 0.92,
        },
        evidence_ids=["model_supplied_id"],
    )

    report = make_report([claim])

    # Simulate output from the correlation analysis tool.
    tool_message = ToolMessage(
        content=json.dumps(
            {
                "dataset_name": "sales.csv",
                "column_a": "revenue",
                "column_b": "quantity",
                "method": "pearson",
                "correlation": 0.92,
                "strength": "strong",
                "direction": "positive",
                "valid_pairs": 9,
                "excluded_rows": 1,
                "interpretation": (
                    "Revenue and quantity have a strong positive correlation."
                ),
            }
        ),
        tool_call_id="test_correlation_tool_call",
        name="analyze_correlation_tool",
    )

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": report,
        "messages": [tool_message],
    }

    with patch(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        return_value=fake_agent,
    ):
        result = investigate_node(
            {
                "user_question": (
                    "What is the correlation between revenue and quantity?"
                ),
                "csv_path": "sales.csv",
                "profile_result": "",
                "steps_taken": 0,
                "errors": [],
            }
        )

    output = json.loads(result["investigation_result"])

    assert len(output["validation_results"]) == 1
    validation = output["validation_results"][0]

    assert validation["claim_id"] == "claim_correlation_1"
    assert validation["status"] == "verified"
    assert validation["evidence_ids"] == ["evidence_1"]


def test_investigate_node_rejects_incorrect_correlation_coefficient():
    claim = FactualClaim(
        claim_id="claim_wrong_correlation",
        statement="Revenue and quantity have a correlation of 0.75.",
        claim_type="correlation",
        claim_data={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
            "correlation": 0.75,
        },
        evidence_ids=["model_supplied_id"],
    )

    report = make_report([claim])

    tool_message = ToolMessage(
        content=json.dumps(
            {
                "dataset_name": "sales.csv",
                "column_a": "revenue",
                "column_b": "quantity",
                "method": "pearson",
                "correlation": 0.92,
                "strength": "strong",
                "direction": "positive",
                "valid_pairs": 9,
                "excluded_rows": 1,
                "interpretation": "Strong positive correlation.",
            }
        ),
        tool_call_id="test_correlation_tool_call",
        name="analyze_correlation_tool",
    )

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": report,
        "messages": [tool_message],
    }

    with patch(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        return_value=fake_agent,
    ):
        result = investigate_node(
            {
                "user_question": "What is the correlation between revenue and quantity?",
                "csv_path": "sales.csv",
                "profile_result": "",
                "steps_taken": 0,
                "errors": [],
            }
        )

    output = json.loads(result["investigation_result"])
    validation = output["validation_results"][0]

    assert validation["claim_id"] == "claim_wrong_correlation"
    assert validation["status"] == "unverifiable"
    assert "evidence_ids" not in validation


def test_investigate_node_rejects_correlation_with_mismatched_columns():
    claim = FactualClaim(
        claim_id="claim_wrong_columns",
        statement="Revenue and quantity have a correlation of 0.92.",
        claim_type="correlation",
        claim_data={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
            "correlation": 0.92,
        },
        evidence_ids=["model_supplied_id"],
    )

    report = make_report([claim])

    tool_message = ToolMessage(
        content=json.dumps(
            {
                "dataset_name": "sales.csv",
                "column_a": "revenue",
                "column_b": "profit",
                "method": "pearson",
                "correlation": 0.92,
                "strength": "strong",
                "direction": "positive",
                "valid_pairs": 9,
                "excluded_rows": 1,
                "interpretation": "Strong positive correlation.",
            }
        ),
        tool_call_id="test_correlation_tool_call",
        name="analyze_correlation_tool",
    )

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": report,
        "messages": [tool_message],
    }

    with patch(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        return_value=fake_agent,
    ):
        result = investigate_node(
            {
                "user_question": "What is the correlation between revenue and quantity?",
                "csv_path": "sales.csv",
                "profile_result": "",
                "steps_taken": 0,
                "errors": [],
            }
        )

    output = json.loads(result["investigation_result"])
    validation = output["validation_results"][0]

    assert validation["claim_id"] == "claim_wrong_columns"
    assert validation["status"] == "unverifiable"
    assert validation["evidence_ids"] == []
