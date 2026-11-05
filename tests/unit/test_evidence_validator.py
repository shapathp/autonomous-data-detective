import pytest

from autonomous_data_detective.agents.evidence_validator import validate_claim
from autonomous_data_detective.agents.schemas import (
    EvidenceRecord,
    FactualClaim,
)



@pytest.fixture
def group_evidence():
    return EvidenceRecord(
        evidence_id="evidence_1",
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



def make_group_claim(group="Engineering", value=95000):
    return FactualClaim(
        claim_id="claim_1",
        statement=f"{group} has an average salary of {value}.",
        claim_type="group_comparison",
        claim_data={
            "group_column": "department",
            "metric_column": "salary",
            "aggregation": "mean",
            "group": group,
            "value": value,
        },
        evidence_ids=["evidence_1"],
    )


def test_valid_highest_group_claim_is_verified(group_evidence):
    result = validate_claim(
        make_group_claim(),
        [group_evidence],
    )

    assert result["status"] == "verified"
    assert result["claim_id"] == "claim_1"


def test_matching_group_not_established_as_highest_is_supported(
    group_evidence,
):
    result = validate_claim(
        make_group_claim(group="Marketing", value=78000),
        [group_evidence],
    )

    assert result["status"] == "supported"



def test_incorrect_group_value_is_unverifiable(group_evidence):
    result = validate_claim(
        make_group_claim(group="Engineering", value=100000),
        [group_evidence],
    )

    assert result["status"] == "unverifiable"



def test_missing_evidence_reference_is_unverifiable():
    claim = FactualClaim(
        claim_id="claim_2",
        statement="Revenue was 50000.",
        claim_type="numerical_result",
        claim_data={"value": 50000},
        evidence_ids=["missing_evidence"],
    )

    result = validate_claim(claim, [])

    assert result["status"] == "unverifiable"
    assert "missing_evidence" in result["reason"]



def test_unsupported_claim_type_is_unverifiable(group_evidence):
    claim = FactualClaim(
        claim_id="claim_3",
        statement="The dataset has a seasonal pattern.",
        claim_type="interpretation",
        claim_data={"interpretation": "seasonal pattern"},
        evidence_ids=["evidence_1"],
    )

    result = validate_claim(claim, [group_evidence])

    assert result["status"] == "unverifiable"



def test_numerical_result_matches_evidence():
    evidence = EvidenceRecord(
        evidence_id="evidence_2",
        tool_name="get_column_statistics_tool",
        tool_output={"mean": 42.0},
    )

    claim = FactualClaim(
        claim_id="claim_4",
        statement="The mean is 42.",
        claim_type="numerical_result",
        claim_data={
            "value": 42.0,
            "result_key": "mean",
        },
        evidence_ids=["evidence_2"],
    )

    result = validate_claim(claim, [evidence])

    assert result["status"] == "verified"



def test_numerical_result_with_wrong_value_is_unverifiable():
    evidence = EvidenceRecord(
        evidence_id="evidence_3",
        tool_name="get_column_statistics_tool",
        tool_output={"mean": 42.0},
    )

    claim = FactualClaim(
        claim_id="claim_5",
        statement="The mean is 50.",
        claim_type="numerical_result",
        claim_data={
            "value": 50.0,
            "result_key": "mean",
        },
        evidence_ids=["evidence_3"],
    )

    result = validate_claim(claim, [evidence])

    assert result["status"] == "unverifiable"
    assert "did not establish" in result["reason"]



def test_numerical_claim_with_wrong_column_is_unverifiable():
    evidence = EvidenceRecord(
        evidence_id="evidence_revenue",
        tool_name="get_column_statistics_tool",
        tool_output={
            "column_name": "quantity",
            "mean": 42.0,
        },
    )

    claim = FactualClaim(
        claim_id="claim_revenue",
        statement="The mean revenue is 50000.",
        claim_type="numerical_result",
        claim_data={
            "column_name": "revenue",
            "value": 50000.0,
            "result_key": "mean",
        },
        evidence_ids=["evidence_revenue"],
    )

    result = validate_claim(claim, [evidence])

    assert result["status"] == "unverifiable"


def test_numerical_claim_with_wrong_aggregation_is_unverifiable():
    evidence = EvidenceRecord(
        evidence_id="evidence_revenue",
        tool_name="aggregate_tool",
        tool_output={
            "column_name": "revenue",
            "aggregation": "sum",
            "sum": 50000.0,
            "mean": 10000.0,
        },
    )

    claim = FactualClaim(
        claim_id="claim_revenue_mean",
        statement="The mean revenue is 50000.",
        claim_type="numerical_result",
        claim_data={
            "column_name": "revenue",
            "aggregation": "mean",
            "value": 50000.0,
            "result_key": "mean",
        },
        evidence_ids=["evidence_revenue"],
    )

    result = validate_claim(claim, [evidence])

    assert result["status"] == "unverifiable"



def test_numerical_result_matches_nested_statistics():
    evidence = EvidenceRecord(
        evidence_id="evidence_nested_stats",
        tool_name="get_column_statistics_tool",
        tool_output={
            "dataset_name": "sales.csv",
            "column_name": "revenue",
            "statistics": {
                "mean": 42000.0,
                "median": 40000.0,
                "min": 10000.0,
                "max": 80000.0,
                "std": 12000.0,
            },
        },
    )

    claim = FactualClaim(
        claim_id="claim_nested_stats",
        statement="The mean revenue is 42000.",
        claim_type="numerical_result",
        claim_data={
            "column_name": "revenue",
            "value": 42000.0,
            "result_key": "mean",
        },
        evidence_ids=["evidence_nested_stats"],
    )

    result = validate_claim(claim, [evidence])

    assert result["status"] == "verified"
    assert result["evidence_ids"] == ["evidence_nested_stats"]



def test_group_claim_with_wrong_metric_is_unverifiable():
    evidence = EvidenceRecord(
        evidence_id="evidence_revenue",
        tool_name="compare_groups_tool",
        tool_output={
            "group_by": "region",
            "metric": "quantity",
            "aggregation": "mean",
            "results": [
                {"group": "North", "value": 90.0},
                {"group": "South", "value": 50.0},
            ],
        },
    )

    claim = FactualClaim(
        claim_id="claim_revenue",
        statement="North has the highest average revenue.",
        claim_type="group_comparison",
        claim_data={
            "group_column": "region",
            "metric_column": "revenue",
            "aggregation": "mean",
            "group": "North",
            "value": 90.0,
        },
        evidence_ids=["evidence_revenue"],
    )

    result = validate_claim(claim, [evidence])

    assert result["status"] == "unverifiable"


def test_group_claim_with_wrong_aggregation_is_unverifiable():
    evidence = EvidenceRecord(
        evidence_id="evidence_revenue",
        tool_name="compare_groups_tool",
        tool_output={
            "group_by": "region",
            "metric": "revenue",
            "aggregation": "sum",
            "results": [
                {"group": "North", "value": 90000.0},
                {"group": "South", "value": 50000.0},
            ],
        },
    )

    claim = FactualClaim(
        claim_id="claim_mean_revenue",
        statement="North has the highest average revenue.",
        claim_type="group_comparison",
        claim_data={
            "group_column": "region",
            "metric_column": "revenue",
            "aggregation": "mean",
            "group": "North",
            "value": 90000.0,
        },
        evidence_ids=["evidence_revenue"],
    )

    result = validate_claim(claim, [evidence])

    assert result["status"] == "unverifiable"



def test_correlation_claim_with_matching_coefficient_is_verified():
    evidence = EvidenceRecord(
        evidence_id="evidence_corr_1",
        tool_name="analyze_correlation_tool",
        tool_output={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
            "correlation": -0.4628,
        },
    )

    claim = FactualClaim(
        claim_id="claim_corr_1",
        statement="Revenue and quantity have a correlation of -0.4628.",
        claim_type="correlation",
        claim_data={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
            "correlation": -0.4628,
        },
        evidence_ids=["evidence_corr_1"],
    )

    result = validate_claim(claim, [evidence])

    assert result["status"] == "verified"
    assert result["evidence_ids"] == ["evidence_corr_1"]


def test_correlation_claim_with_wrong_coefficient_is_unverifiable():
    evidence = EvidenceRecord(
        evidence_id="evidence_corr_2",
        tool_name="analyze_correlation_tool",
        tool_output={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
            "correlation": -0.4628,
        },
    )

    claim = FactualClaim(
        claim_id="claim_corr_2",
        statement="Revenue and quantity have a correlation of 0.85.",
        claim_type="correlation",
        claim_data={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
            "correlation": 0.85,
        },
        evidence_ids=["evidence_corr_2"],
    )

    result = validate_claim(claim, [evidence])

    assert result["status"] == "unverifiable"


def test_correlation_claim_with_wrong_method_is_unverifiable():
    evidence = EvidenceRecord(
        evidence_id="evidence_corr_3",
        tool_name="analyze_correlation_tool",
        tool_output={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
            "correlation": -0.4628,
        },
    )

    claim = FactualClaim(
        claim_id="claim_corr_3",
        statement="Revenue and quantity have a Spearman correlation of -0.4628.",
        claim_type="correlation",
        claim_data={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "spearman",
            "correlation": -0.4628,
        },
        evidence_ids=["evidence_corr_3"],
    )

    result = validate_claim(claim, [evidence])

    assert result["status"] == "unverifiable"



def test_numerical_claim_without_expected_value_is_unverifiable():
    evidence = EvidenceRecord(
        evidence_id="evidence_missing_expected_value",
        tool_name="get_column_statistics_tool",
        tool_output={"column_name": "revenue", "mean": 42.0},
    )

    claim = FactualClaim(
        claim_id="claim_missing_value",
        statement="The mean revenue is 42.",
        claim_type="numerical_result",
        claim_data={
            "column_name": "revenue",
            "result_key": "mean",
        },
        evidence_ids=["evidence_missing_expected_value"],
    )

    result = validate_claim(claim, [evidence])

    assert result["status"] == "unverifiable"


def test_correlation_claim_without_column_metadata_is_unverifiable():
    evidence = EvidenceRecord(
        evidence_id="evidence_corr_missing_columns",
        tool_name="analyze_correlation_tool",
        tool_output={
            "column_a": "revenue",
            "column_b": "quantity",
            "method": "pearson",
            "correlation": 0.72,
        },
    )

    claim = FactualClaim(
        claim_id="claim_corr_missing_columns",
        statement="The correlation coefficient is 0.72.",
        claim_type="correlation",
        claim_data={
            "method": "pearson",
            "correlation": 0.72,
        },
        evidence_ids=["evidence_corr_missing_columns"],
    )

    result = validate_claim(claim, [evidence])

    assert result["status"] == "unverifiable"



def test_group_claim_without_required_metadata_is_unverifiable(
    group_evidence,
):
    claim = FactualClaim(
        claim_id="claim_missing_group_metadata",
        statement="Engineering has an average salary of 95000.",
        claim_type="group_comparison",
        claim_data={
            "aggregation": "mean",
            "group": "Engineering",
            "value": 95000,
        },
        evidence_ids=["evidence_1"],
    )

    result = validate_claim(claim, [group_evidence])

    assert result["status"] == "unverifiable"


def test_group_claim_does_not_trust_first_result_when_results_are_unsorted(
    group_evidence,
):
    from copy import deepcopy

    evidence = deepcopy(group_evidence)

    evidence.tool_output["results"] = [
        {"group": "Engineering", "value": 95000},
        {"group": "Sales", "value": 120000},
        {"group": "Support", "value": 70000},
    ]

    claim = FactualClaim(
        claim_id="claim_unsorted_group_results",
        statement="Engineering has the highest average salary of 95000.",
        claim_type="group_comparison",
        claim_data={
            "group_column": "department",
            "metric_column": "salary",
            "aggregation": "mean",
            "group": "Engineering",
            "value": 95000,
        },
        evidence_ids=["evidence_1"],
    )

    result = validate_claim(claim, [evidence])

    assert result["status"] != "verified"
