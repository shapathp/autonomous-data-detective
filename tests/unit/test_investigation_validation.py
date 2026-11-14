import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import ToolMessage

from autonomous_data_detective.agents.schemas import (
    FactualClaim,
    InvestigationReport,
)
from autonomous_data_detective.graph.nodes import (
    _get_highest_groups,
    investigate_node,
)


def test_get_highest_groups_does_not_duplicate_tied_groups():
    tool_output = {
        "highest_groups": ["North"],
        "results": [
            {"group": "North", "value": 100},
            {"group": "North", "value": 100},
            {"group": "South", "value": 80},
        ],
    }

    highest_groups, maximum = _get_highest_groups(tool_output)

    assert highest_groups == ["North"]
    assert maximum == 100



def test_get_highest_groups_ignores_conflicting_explicit_groups():
    tool_output = {
        "highest_groups": ["North", "South"],
        "results": [
            {"group": "North", "value": 100},
            {"group": "South", "value": 90},
            {"group": "West", "value": 80},
        ],
    }

    highest_groups, maximum = _get_highest_groups(tool_output)

    assert highest_groups == ["North"]
    assert maximum == 100

def test_get_highest_groups_prefers_numeric_results_over_conflicting_metadata():
    tool_output = {
        "highest_groups": ["North", "South"],
        "results": [
            {"group": "North", "value": 100},
            {"group": "South", "value": 90},
            {"group": "West", "value": 80},
        ],
    }

    highest_groups, maximum = _get_highest_groups(tool_output)

    assert highest_groups == ["North"]
    assert maximum == 100

def test_get_highest_groups_does_not_trust_metadata_without_valid_results():
    tool_output = {
        "highest_groups": ["North", "South"],
        "results": [
            {"group": "North", "value": float("nan")},
            {"group": "South", "value": float("inf")},
        ],
    }

    highest_groups, maximum = _get_highest_groups(tool_output)

    assert highest_groups == ["North", "South"]
    assert maximum is None


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

    print("DEBUG investigate_node result:", result)
    output = json.loads(result["investigation_result"])

    assert len(output["validation_results"]) == 1
    validation = output["validation_results"][0]

    assert validation["claim_id"] == "claim_1"
    assert validation["status"] == "verified"
    assert validation["evidence_ids"] == ["evidence_1"]
    assert output["claims"][0]["evidence_ids"] == ["evidence_1"]


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
    assert output["claims"][0]["evidence_ids"] == []
    assert output["summary"] == report.summary
    assert output["findings"] == report.findings



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


def test_investigate_node_reports_all_tied_highest_groups():
    claim = FactualClaim(
        claim_id="claim_tied_winners",
        statement="Engineering had the uniquely highest average salary.",
        claim_type="group_comparison",
        claim_data={
            "group": "Engineering",
            "value": 95000,
            "group_column": "department",
            "metric_column": "salary",
            "aggregation": "mean",
        },
        evidence_ids=["model_supplied_id"],
    )

    report = InvestigationReport(
        summary="Engineering had the highest average salary.",
        findings=["Engineering was the unique salary leader."],
        evidence=["Group comparison tool output."],
        limitations=[],
        recommendations=[],
        claims=[claim],
    )

    tool_message = make_tool_message(
        {
            "group_by": "department",
            "metric": "salary",
            "aggregation": "mean",
            "highest_group": "Engineering",
            "highest_value": 95000,
            "highest_groups": ["Engineering", "Marketing"],
            "results": [
                {"group": "Engineering", "value": 95000},
                {"group": "Marketing", "value": 95000},
                {"group": "Sales", "value": 70000},
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
                "user_question": (
                    "Which department has the highest average salary?"
                ),
                "csv_path": "salaries.csv",
                "profile_result": "",
                "steps_taken": 0,
                "errors": [],
            }
        )

    output = json.loads(result["investigation_result"])

    assert "Engineering" in output["summary"]
    assert "Marketing" in output["summary"]
    assert "95000" in output["summary"]

    assert any(
        "Engineering" in finding
        and "Marketing" in finding
        and "95000" in finding
        for finding in output["findings"]
    )


def test_investigate_node_preserves_report_for_unique_winner():
    claim = FactualClaim(
        claim_id="claim_unique_winner",
        statement="Engineering had the highest average salary.",
        claim_type="group_comparison",
        claim_data={
            "group": "Engineering",
            "value": 95000,
            "group_column": "department",
            "metric_column": "salary",
            "aggregation": "mean",
        },
        evidence_ids=["model_supplied_id"],
    )

    report = InvestigationReport(
        summary="Engineering had the highest average salary.",
        findings=[
            "Engineering had the highest average salary."
        ],
        evidence=["Group comparison tool output."],
        limitations=[],
        recommendations=[],
        claims=[claim],
    )

    tool_message = make_tool_message(
        {
            "group_by": "department",
            "metric": "salary",
            "aggregation": "mean",
            "highest_group": "Engineering",
            "highest_value": 95000,
            "highest_groups": ["Engineering"],
            "results": [
                {"group": "Engineering", "value": 95000},
                {"group": "Marketing", "value": 85000},
                {"group": "Sales", "value": 70000},
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
                "user_question": (
                    "Which department has the highest average salary?"
                ),
                "csv_path": "salaries.csv",
                "profile_result": "",
                "steps_taken": 0,
                "errors": [],
            }
        )

    output = json.loads(result["investigation_result"])

    assert output["summary"] == (
        "Engineering had the highest average salary."
    )
    assert output["findings"] == [
        "Engineering had the highest average salary."
    ]


def test_investigate_node_preserves_unrelated_findings_when_tie_exists():
    salary_claim = FactualClaim(
        claim_id="claim_salary_tie",
        statement="Engineering had the uniquely highest average salary.",
        claim_type="group_comparison",
        claim_data={
            "group": "Engineering",
            "value": 95000,
            "group_column": "department",
            "metric_column": "salary",
            "aggregation": "mean",
        },
        evidence_ids=["model_salary"],
    )

    revenue_claim = FactualClaim(
        claim_id="claim_revenue",
        statement="Retail had the highest average revenue.",
        claim_type="group_comparison",
        claim_data={
            "group": "Retail",
            "value": 80000,
            "group_column": "store_type",
            "metric_column": "revenue",
            "aggregation": "mean",
        },
        evidence_ids=["model_revenue"],
    )

    report = InvestigationReport(
        summary="Engineering had the highest average salary.",
        findings=[
            "Engineering was the unique salary leader.",
            "Retail had the highest average revenue.",
        ],
        evidence=[
            "Salary comparison output.",
            "Revenue comparison output.",
        ],
        limitations=[],
        recommendations=[],
        claims=[salary_claim, revenue_claim],
    )

    salary_message = make_tool_message(
        {
            "group_by": "department",
            "metric": "salary",
            "aggregation": "mean",
            "highest_group": "Engineering",
            "highest_value": 95000,
            "highest_groups": ["Engineering", "Marketing"],
            "results": [
                {"group": "Engineering", "value": 95000},
                {"group": "Marketing", "value": 95000},
                {"group": "Sales", "value": 70000},
            ],
        }
    )

    revenue_message = ToolMessage(
        content=json.dumps(
            {
                "group_by": "store_type",
                "metric": "revenue",
                "aggregation": "mean",
                "highest_group": "Retail",
                "highest_value": 80000,
                "highest_groups": ["Retail"],
                "results": [
                    {"group": "Retail", "value": 80000},
                    {"group": "Wholesale", "value": 65000},
                ],
            }
        ),
        tool_call_id="revenue_tool_call",
        name="compare_groups_tool",
    )

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": report,
        "messages": [salary_message, revenue_message],
    }

    with patch(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        return_value=fake_agent,
    ):
        result = investigate_node(
            {
                "user_question": (
                    "Compare average salary by department and "
                    "average revenue by store type."
                ),
                "csv_path": "business.csv",
                "profile_result": "",
                "steps_taken": 0,
                "errors": [],
            }
        )

    output = json.loads(result["investigation_result"])

    assert "Engineering" in output["summary"]
    assert "Marketing" in output["summary"]
    assert "95000" in output["summary"]

    assert any(
        "Retail" in finding and "revenue" in finding.lower()
        for finding in output["findings"]
    )



def test_investigate_node_reports_ties_in_multiple_comparisons():
    salary_claim = FactualClaim(
        claim_id="claim_salary_tie",
        statement="Engineering had the uniquely highest average salary.",
        claim_type="group_comparison",
        claim_data={
            "group": "Engineering",
            "value": 95000,
            "group_column": "department",
            "metric_column": "salary",
            "aggregation": "mean",
        },
        evidence_ids=["model_salary"],
    )

    revenue_claim = FactualClaim(
        claim_id="claim_revenue_tie",
        statement="Retail had the uniquely highest average revenue.",
        claim_type="group_comparison",
        claim_data={
            "group": "Retail",
            "value": 80000,
            "group_column": "store_type",
            "metric_column": "revenue",
            "aggregation": "mean",
        },
        evidence_ids=["model_revenue"],
    )

    report = InvestigationReport(
        summary="Engineering had the highest average salary.",
        findings=[
            "Engineering was the unique salary leader.",
            "Retail was the unique revenue leader.",
        ],
        evidence=[
            "Salary comparison output.",
            "Revenue comparison output.",
        ],
        limitations=[],
        recommendations=[],
        claims=[salary_claim, revenue_claim],
    )

    salary_message = make_tool_message(
        {
            "group_by": "department",
            "metric": "salary",
            "aggregation": "mean",
            "highest_group": "Engineering",
            "highest_value": 95000,
            "highest_groups": ["Engineering", "Marketing"],
            "results": [
                {"group": "Engineering", "value": 95000},
                {"group": "Marketing", "value": 95000},
                {"group": "Sales", "value": 70000},
            ],
        }
    )

    revenue_message = ToolMessage(
        content=json.dumps(
            {
                "group_by": "store_type",
                "metric": "revenue",
                "aggregation": "mean",
                "highest_group": "Retail",
                "highest_value": 80000,
                "highest_groups": ["Retail", "Online"],
                "results": [
                    {"group": "Retail", "value": 80000},
                    {"group": "Online", "value": 80000},
                    {"group": "Wholesale", "value": 65000},
                ],
            }
        ),
        tool_call_id="revenue_tool_call",
        name="compare_groups_tool",
    )

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": report,
        "messages": [salary_message, revenue_message],
    }

    with patch(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        return_value=fake_agent,
    ):
        result = investigate_node(
            {
                "user_question": (
                    "Compare average salary by department and "
                    "average revenue by store type."
                ),
                "csv_path": "business.csv",
                "profile_result": "",
                "steps_taken": 0,
                "errors": [],
            }
        )

    output = json.loads(result["investigation_result"])

    # Both independent ties must appear in the summary.
    assert "Engineering" in output["summary"]
    assert "Marketing" in output["summary"]
    assert "95000" in output["summary"]

    assert "Retail" in output["summary"]
    assert "Online" in output["summary"]
    assert "80000" in output["summary"]

    # Both ties must also appear in the findings.
    assert any(
        "Engineering" in finding
        and "Marketing" in finding
        and "95000" in finding
        for finding in output["findings"]
    )

    assert any(
        "Retail" in finding
        and "Online" in finding
        and "80000" in finding
        for finding in output["findings"]
    )



def test_investigate_node_preserves_finding_when_group_name_overlaps_tie():
    salary_claim = FactualClaim(
        claim_id="claim_salary_tie",
        statement="Engineering had the uniquely highest average salary.",
        claim_type="group_comparison",
        claim_data={
            "group": "Engineering",
            "value": 95000,
            "group_column": "department",
            "metric_column": "salary",
            "aggregation": "mean",
        },
        evidence_ids=["model_salary"],
    )

    revenue_claim = FactualClaim(
        claim_id="claim_revenue",
        statement="Engineering had the highest average revenue.",
        claim_type="group_comparison",
        claim_data={
            "group": "Engineering",
            "value": 80000,
            "group_column": "department",
            "metric_column": "revenue",
            "aggregation": "mean",
        },
        evidence_ids=["model_revenue"],
    )

    report = InvestigationReport(
        summary="Engineering had the highest average salary.",
        findings=[
            "Engineering was the unique salary leader.",
            "Engineering had the highest average revenue.",
        ],
        evidence=["Salary comparison.", "Revenue comparison."],
        limitations=[],
        recommendations=[],
        claims=[salary_claim, revenue_claim],
    )

    salary_message = make_tool_message(
        {
            "group_by": "department",
            "metric": "salary",
            "aggregation": "mean",
            "highest_group": "Engineering",
            "highest_value": 95000,
            "highest_groups": ["Engineering", "Marketing"],
            "results": [
                {"group": "Engineering", "value": 95000},
                {"group": "Marketing", "value": 95000},
                {"group": "Sales", "value": 70000},
            ],
        }
    )

    revenue_message = ToolMessage(
        content=json.dumps(
            {
                "group_by": "department",
                "metric": "revenue",
                "aggregation": "mean",
                "highest_group": "Engineering",
                "highest_value": 80000,
                "highest_groups": ["Engineering"],
                "results": [
                    {"group": "Engineering", "value": 80000},
                    {"group": "Marketing", "value": 75000},
                    {"group": "Sales", "value": 60000},
                ],
            }
        ),
        tool_call_id="revenue_tool_call",
        name="compare_groups_tool",
    )

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": report,
        "messages": [salary_message, revenue_message],
    }

    with patch(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        return_value=fake_agent,
    ):
        result = investigate_node(
            {
                "user_question": (
                    "Compare average salary and average revenue "
                    "by department."
                ),
                "csv_path": "business.csv",
                "profile_result": "",
                "steps_taken": 0,
                "errors": [],
            }
        )

    output = json.loads(result["investigation_result"])

    assert any(
        "Engineering" in finding
        and "revenue" in finding.lower()
        and "highest" in finding.lower()
        for finding in output["findings"]
    )


def test_investigate_node_preserves_unique_winner_for_unrelated_metric():
    salary_claim = FactualClaim(
        claim_id="claim_salary_tie",
        statement="Engineering had the uniquely highest average salary.",
        claim_type="group_comparison",
        claim_data={
            "group": "Engineering",
            "value": 95000,
            "group_column": "department",
            "metric_column": "salary",
            "aggregation": "mean",
        },
        evidence_ids=["model_salary"],
    )

    revenue_claim = FactualClaim(
        claim_id="claim_revenue",
        statement="Retail was the unique leader for revenue.",
        claim_type="group_comparison",
        claim_data={
            "group": "Retail",
            "value": 80000,
            "group_column": "store_type",
            "metric_column": "revenue",
            "aggregation": "mean",
        },
        evidence_ids=["model_revenue"],
    )

    report = InvestigationReport(
        summary="Engineering had the highest average salary.",
        findings=[
            "Engineering was the unique salary leader.",
            "Retail was the unique leader for revenue.",
        ],
        evidence=[
            "Salary comparison output.",
            "Revenue comparison output.",
        ],
        limitations=[],
        recommendations=[],
        claims=[salary_claim, revenue_claim],
    )

    salary_message = make_tool_message(
        {
            "group_by": "department",
            "metric": "salary",
            "aggregation": "mean",
            "highest_group": "Engineering",
            "highest_value": 95000,
            "highest_groups": ["Engineering", "Marketing"],
            "results": [
                {"group": "Engineering", "value": 95000},
                {"group": "Marketing", "value": 95000},
                {"group": "Sales", "value": 70000},
            ],
        }
    )

    revenue_message = ToolMessage(
        content=json.dumps(
            {
                "group_by": "store_type",
                "metric": "revenue",
                "aggregation": "mean",
                "highest_group": "Retail",
                "highest_value": 80000,
                "highest_groups": ["Retail"],
                "results": [
                    {"group": "Retail", "value": 80000},
                    {"group": "Wholesale", "value": 65000},
                ],
            }
        ),
        tool_call_id="revenue_tool_call",
        name="compare_groups_tool",
    )

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": report,
        "messages": [salary_message, revenue_message],
    }

    with patch(
        "autonomous_data_detective.graph.nodes.build_investigator_agent",
        return_value=fake_agent,
    ):
        result = investigate_node(
            {
                "user_question": (
                    "Compare average salary by department and "
                    "average revenue by store type."
                ),
                "csv_path": "business.csv",
                "profile_result": "",
                "steps_taken": 0,
                "errors": [],
            }
        )

    output = json.loads(result["investigation_result"])

    assert "Engineering" in output["summary"]
    assert "Marketing" in output["summary"]
    assert "95000" in output["summary"]

    assert (
        "Retail was the unique leader for revenue."
        in output["findings"]
    )


def test_investigate_node_does_not_report_tie_from_invalid_numeric_results():
    claim = FactualClaim(
        claim_id="claim_invalid_tie",
        statement="Engineering had the uniquely highest average salary.",
        claim_type="group_comparison",
        claim_data={
            "group": "Engineering",
            "value": 95000,
            "group_column": "department",
            "metric_column": "salary",
            "aggregation": "mean",
        },
        evidence_ids=["model_supplied_id"],
    )

    report = InvestigationReport(
        summary="Engineering had the highest average salary.",
        findings=["Engineering was the unique salary leader."],
        evidence=["Group comparison tool output."],
        limitations=[],
        recommendations=[],
        claims=[claim],
    )

    tool_message = make_tool_message(
        {
            "group_by": "department",
            "metric": "salary",
            "aggregation": "mean",
            "highest_groups": ["Engineering", "Marketing"],
            "results": [
                {"group": "Engineering", "value": float("nan")},
                {"group": "Marketing", "value": float("inf")},
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
                "user_question": "Which department has the highest average salary?",
                "csv_path": "salaries.csv",
                "profile_result": "",
                "steps_taken": 0,
                "errors": [],
            }
        )

    output = json.loads(result["investigation_result"])

    assert output["summary"] == (
        "Engineering had the highest average salary."
    )
    assert output["findings"] == [
        "Engineering was the unique salary leader."
    ]
