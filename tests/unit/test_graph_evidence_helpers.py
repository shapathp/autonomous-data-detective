import json

from langchain_core.messages import HumanMessage, ToolMessage

from autonomous_data_detective.agents.schemas import (
    EvidenceRecord,
    FactualClaim,
)
from autonomous_data_detective.graph.nodes import (
    extract_tool_evidence,
    find_evidence_for_claim,
)


def test_extract_tool_evidence_accepts_valid_tool_output():
    message = ToolMessage(
        content=json.dumps({"column_name": "revenue", "mean": 100}),
        tool_call_id="call_1",
        name="get_column_statistics_tool",
    )

    result = extract_tool_evidence([message])

    assert len(result) == 1
    assert result[0].evidence_id == "evidence_1"
    assert result[0].tool_name == "get_column_statistics_tool"
    assert result[0].tool_output["mean"] == 100


def test_extract_tool_evidence_ignores_non_tool_messages():
    result = extract_tool_evidence(
        [HumanMessage(content="Analyze the dataset")]
    )

    assert result == []


def test_extract_tool_evidence_ignores_invalid_json():
    message = ToolMessage(
        content="not valid JSON",
        tool_call_id="call_1",
        name="get_column_statistics_tool",
    )

    assert extract_tool_evidence([message]) == []


def test_extract_tool_evidence_ignores_json_that_is_not_a_dictionary():
    message = ToolMessage(
        content=json.dumps([1, 2, 3]),
        tool_call_id="call_1",
        name="get_column_statistics_tool",
    )

    assert extract_tool_evidence([message]) == []


def test_extract_tool_evidence_numbers_only_accepted_records():
    messages = [
        ToolMessage(
            content="invalid JSON",
            tool_call_id="call_1",
            name="first_tool",
        ),
        ToolMessage(
            content=json.dumps({"value": 42}),
            tool_call_id="call_2",
            name="second_tool",
        ),
    ]

    result = extract_tool_evidence(messages)

    assert len(result) == 1
    assert result[0].evidence_id == "evidence_1"
    assert result[0].tool_name == "second_tool"


def test_find_evidence_matches_numerical_result():
    claim = FactualClaim(
        claim_id="claim_1",
        statement="Revenue mean is 100.",
        claim_type="numerical_result",
        claim_data={
            "column_name": "revenue",
            "result_key": "mean",
            "value": 100,
        },
        evidence_ids=["model_supplied_id"],
    )

    evidence = EvidenceRecord(
        evidence_id="evidence_1",
        tool_name="get_column_statistics_tool",
        tool_output={
            "column_name": "revenue",
            "statistics": {"mean": 100},
        },
    )

    result = find_evidence_for_claim(claim, [evidence])

    assert result == [evidence]


def test_find_evidence_matches_numerical_result_at_top_level():
    claim = FactualClaim(
        claim_id="claim_2",
        statement="Revenue mean is 100.",
        claim_type="numerical_result",
        claim_data={
            "column_name": "revenue",
            "result_key": "mean",
            "value": 100,
        },
        evidence_ids=["model_supplied_id"],
    )

    evidence = EvidenceRecord(
        evidence_id="evidence_2",
        tool_name="get_column_statistics_tool",
        tool_output={
            "column_name": "revenue",
            "mean": 100,
        },
    )

    result = find_evidence_for_claim(claim, [evidence])

    assert result == [evidence]


def test_find_evidence_rejects_correlation_without_coefficient():
    claim = FactualClaim(
        claim_id="claim_3",
        statement="Revenue and quantity are correlated.",
        claim_type="correlation",
        claim_data={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
        },
        evidence_ids=["model_supplied_id"],
    )

    evidence = EvidenceRecord(
        evidence_id="evidence_3",
        tool_name="analyze_correlation_tool",
        tool_output={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
            "correlation": None,
        },
    )

    assert find_evidence_for_claim(claim, [evidence]) == []


def test_find_evidence_rejects_mismatched_correlation_method():
    claim = FactualClaim(
        claim_id="claim_4",
        statement="Revenue and quantity have a positive correlation.",
        claim_type="correlation",
        claim_data={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
        },
        evidence_ids=["model_supplied_id"],
    )

    evidence = EvidenceRecord(
        evidence_id="evidence_4",
        tool_name="analyze_correlation_tool",
        tool_output={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "spearman",
            "correlation": 0.8,
        },
    )

    assert find_evidence_for_claim(claim, [evidence]) == []



def test_extract_tool_evidence_skips_tool_message_with_non_string_content():
    message = ToolMessage(
        content=[{"type": "text", "text": "not a JSON string"}],
        tool_call_id="call_non_string",
        name="some_tool",
    )

    assert extract_tool_evidence([message]) == []


def test_find_evidence_rejects_group_comparison_from_wrong_tool():
    claim = FactualClaim(
        claim_id="group_1",
        statement="North has the highest mean revenue.",
        claim_type="group_comparison",
        claim_data={
            "group_column": "region",
            "metric_column": "revenue",
            "aggregation": "mean",
        },
        evidence_ids=["model_supplied_id"],
    )
    evidence = EvidenceRecord(
        evidence_id="evidence_1",
        tool_name="get_column_statistics_tool",
        tool_output={
            "group_by": "region",
            "metric": "revenue",
            "aggregation": "mean",
        },
    )

    assert find_evidence_for_claim(claim, [evidence]) == []


def test_find_evidence_rejects_mismatched_group_column():
    claim = FactualClaim(
        claim_id="group_2",
        statement="North has the highest mean revenue.",
        claim_type="group_comparison",
        claim_data={
            "group_column": "region",
            "metric_column": "revenue",
            "aggregation": "mean",
        },
        evidence_ids=["model_supplied_id"],
    )
    evidence = EvidenceRecord(
        evidence_id="evidence_2",
        tool_name="compare_groups_tool",
        tool_output={
            "group_by": "department",
            "metric": "revenue",
            "aggregation": "mean",
        },
    )

    assert find_evidence_for_claim(claim, [evidence]) == []


def test_find_evidence_rejects_mismatched_group_aggregation():
    claim = FactualClaim(
        claim_id="group_3",
        statement="North has the highest mean revenue.",
        claim_type="group_comparison",
        claim_data={
            "group_column": "region",
            "metric_column": "revenue",
            "aggregation": "mean",
        },
        evidence_ids=["model_supplied_id"],
    )
    evidence = EvidenceRecord(
        evidence_id="evidence_3",
        tool_name="compare_groups_tool",
        tool_output={
            "group_by": "region",
            "metric": "revenue",
            "aggregation": "sum",
        },
    )

    assert find_evidence_for_claim(claim, [evidence]) == []


def test_find_evidence_rejects_numerical_result_for_different_column():
    claim = FactualClaim(
        claim_id="numeric_1",
        statement="Revenue mean is 100.",
        claim_type="numerical_result",
        claim_data={
            "column_name": "revenue",
            "result_key": "mean",
        },
        evidence_ids=["model_supplied_id"],
    )
    evidence = EvidenceRecord(
        evidence_id="evidence_4",
        tool_name="get_column_statistics_tool",
        tool_output={
            "column_name": "quantity",
            "statistics": {"mean": 100},
        },
    )

    assert find_evidence_for_claim(claim, [evidence]) == []


def test_find_evidence_rejects_correlation_from_wrong_tool():
    claim = FactualClaim(
        claim_id="correlation_1",
        statement="Revenue and quantity are correlated.",
        claim_type="correlation",
        claim_data={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
        },
        evidence_ids=["model_supplied_id"],
    )
    evidence = EvidenceRecord(
        evidence_id="evidence_5",
        tool_name="compare_groups_tool",
        tool_output={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
            "correlation": 0.8,
        },
    )

    assert find_evidence_for_claim(claim, [evidence]) == []


def test_find_evidence_rejects_mismatched_first_correlation_column():
    claim = FactualClaim(
        claim_id="correlation_2",
        statement="Revenue and quantity are correlated.",
        claim_type="correlation",
        claim_data={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
        },
        evidence_ids=["model_supplied_id"],
    )
    evidence = EvidenceRecord(
        evidence_id="evidence_6",
        tool_name="analyze_correlation_tool",
        tool_output={
            "column_a": "profit",
            "column_b": "quantity",
            "method": "pearson",
            "correlation": 0.8,
        },
    )

    assert find_evidence_for_claim(claim, [evidence]) == []



def test_find_evidence_matches_group_columns_with_different_case_and_whitespace():
    claim = FactualClaim(
        claim_id="group_case_1",
        statement="Engineering has the highest mean salary.",
        claim_type="group_comparison",
        claim_data={
            "group_column": " DEPARTMENT ",
            "metric_column": " SALARY ",
            "aggregation": "mean",
        },
        evidence_ids=["model_supplied_id"],
    )

    evidence = EvidenceRecord(
        evidence_id="evidence_group_case",
        tool_name="compare_groups_tool",
        tool_output={
            "group_by": "department",
            "metric": "salary",
            "aggregation": "mean",
            "highest_group": "Engineering",
            "highest_value": 95000,
            "results": [
                {"group": "Engineering", "value": 95000},
                {"group": "Marketing", "value": 78000},
            ],
        },
    )

    result = find_evidence_for_claim(claim, [evidence])

    assert result == [evidence]


def test_find_evidence_matches_numerical_column_with_different_case_and_whitespace():
    claim = FactualClaim(
        claim_id="numeric_case_1",
        statement="Revenue mean is 42000.",
        claim_type="numerical_result",
        claim_data={
            "column_name": " REVENUE ",
            "result_key": "mean",
            "value": 42000,
        },
        evidence_ids=["model_supplied_id"],
    )

    evidence = EvidenceRecord(
        evidence_id="evidence_numeric_case",
        tool_name="get_column_statistics_tool",
        tool_output={
            "column_name": "Revenue",
            "statistics": {"mean": 42000},
        },
    )

    result = find_evidence_for_claim(claim, [evidence])

    assert result == [evidence]


def test_find_evidence_matches_correlation_columns_with_different_case_and_whitespace():
    claim = FactualClaim(
        claim_id="correlation_case_1",
        statement="Revenue and quantity have a positive correlation.",
        claim_type="correlation",
        claim_data={
            "column_a": " REVENUE ",
            "column_b": " QUANTITY ",
            "method": "pearson",
        },
        evidence_ids=["model_supplied_id"],
    )

    evidence = EvidenceRecord(
        evidence_id="evidence_correlation_case",
        tool_name="analyze_correlation_tool",
        tool_output={
            "column_a": "Revenue",
            "column_b": "Quantity",
            "method": "pearson",
            "correlation": 0.75,
        },
    )

    result = find_evidence_for_claim(claim, [evidence])

    assert result == [evidence]


def test_find_evidence_matches_numerical_result_using_metric_column_and_aggregation():
    claim = FactualClaim(
        claim_id="numeric_metric_column_1",
        statement="The overall mean salary is approximately 86666.67.",
        claim_type="numerical_result",
        claim_data={
            "metric_column": "salary",
            "aggregation": "mean",
            "value": 86666.66666666667,
        },
        evidence_ids=["model_supplied_id"],
    )

    evidence = EvidenceRecord(
        evidence_id="evidence_salary_mean",
        tool_name="get_column_statistics_tool",
        tool_output={
            "column_name": "salary",
            "statistics": {
                "mean": 86666.66666666667,
                "median": 90000.0,
                "std": 12000.0,
                "variance": 144000000.0,
            },
        },
    )

    result = find_evidence_for_claim(claim, [evidence])

    assert result == [evidence]



def test_find_evidence_matches_numerical_claim_using_column_and_metric():
    """Support alternative column and metric keys in numerical claims."""

    claim = FactualClaim(
        claim_id="variance_claim_1",
        statement="The variance of salary is approximately 160625000.",
        claim_type="numerical_result",
        claim_data={
            "column": "salary",
            "metric": "variance",
            "value": 160625000,
        },
        evidence_ids=["evidence_2"],
    )

    evidence = EvidenceRecord(
        evidence_id="evidence_2",
        tool_name="get_column_statistics_tool",
        tool_output={
            "column_name": "salary",
            "statistics": {
                "mean": 86666.66666666667,
                "median": 92500.0,
                "std": 12673.044646258475,
                "variance": 160625000,
            },
        },
    )

    result = find_evidence_for_claim(claim, [evidence])

    assert result == [evidence]

def test_find_evidence_rejects_numerical_claim_without_column_metadata():
    claim = FactualClaim(
        claim_id="numeric_missing_column",
        statement="The mean is 100.",
        claim_type="numerical_result",
        claim_data={
            "result_key": "mean",
            "value": 100,
        },
        evidence_ids=["model_supplied_id"],
    )

    evidence = EvidenceRecord(
        evidence_id="evidence_revenue",
        tool_name="get_column_statistics_tool",
        tool_output={
            "column_name": "revenue",
            "statistics": {"mean": 100},
        },
    )

    assert find_evidence_for_claim(claim, [evidence]) == []