import pytest
from pydantic import ValidationError

from autonomous_data_detective.agents.schemas import (
    EvidenceRecord,
    FactualClaim,
    InvestigationReport,
)


def has_expected_answer(
    report: InvestigationReport,
    expected_region: str,
    expected_value: float,
) -> bool:
    """Check whether a report contains the expected region and value."""

    report_text = " ".join(
        [report.summary, *report.findings]
    ).lower().replace(",", "")

    normalized_value = str(int(expected_value))

    return (
        expected_region.lower() in report_text
        and normalized_value in report_text
    )

def test_investigation_report_accepts_valid_data():
    report = InvestigationReport(
        summary="North has the highest average revenue.",
        findings=["North has the highest average revenue."],
        evidence=["The comparison tool returned North as the highest group."],
        limitations=["One row was excluded because revenue was missing."],
        recommendations=["Investigate why revenue is missing."],
    )

    assert report.summary == "North has the highest average revenue."
    assert len(report.findings) == 1
    assert len(report.evidence) == 1


def test_investigation_report_rejects_invalid_field_type():
    with pytest.raises(ValidationError):
        InvestigationReport(
            summary="Analysis complete.",
            findings="This should be a list",
            evidence=[],
            limitations=[],
            recommendations=[],
        )


def test_incorrect_region_is_not_accepted_as_correct_answer():
    report = InvestigationReport(
        summary="The region with the highest average revenue is West.",
        findings=["West has the highest average revenue, at 50,000."],
        evidence=["The comparison tool returned West as the highest group."],
        limitations=[],
        recommendations=[],
    )

    assert not has_expected_answer(
        report,
        expected_region="North",
        expected_value=56000,
    )



def test_investigation_report_rejects_empty_summary():
    with pytest.raises(ValidationError):
        InvestigationReport(
            summary="",
            findings=["North has the highest average revenue."],
            evidence=["The comparison tool returned North."],
            limitations=[],
            recommendations=[],
        )


def test_investigation_report_rejects_empty_findings():
    with pytest.raises(ValidationError):
        InvestigationReport(
            summary="Analysis complete.",
            findings=[],
            evidence=["The comparison tool returned North."],
            limitations=[],
            recommendations=[],
        )


def test_investigation_report_rejects_empty_evidence():
    with pytest.raises(ValidationError):
        InvestigationReport(
            summary="North has the highest average revenue.",
            findings=["North has the highest average revenue."],
            evidence=[],
            limitations=[],
            recommendations=[],
        )


def test_factual_claim_accepts_group_comparison():
    claim = FactualClaim(
        claim_id="claim_1",
        statement="The Engineering department has the highest average salary.",
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

    assert claim.claim_type == "group_comparison"
    assert claim.claim_data["group_column"] == "department"
    assert claim.claim_data["group"] == "Engineering"
    assert claim.claim_data["value"] == 95000


def test_factual_claim_accepts_correlation():
    claim = FactualClaim(
        claim_id="claim_2",
        statement="Website visits and purchases have a positive correlation.",
        claim_type="correlation",
        claim_data={
            "column_a": "website_visits",
            "column_b": "purchases",
            "method": "pearson",
            "coefficient": 0.72,
        },
        evidence_ids=["evidence_2"],
    )

    assert claim.claim_type == "correlation"
    assert claim.claim_data["coefficient"] == 0.72


def test_evidence_record_accepts_structured_tool_output():
    evidence = EvidenceRecord(
        evidence_id="evidence_1",
        tool_name="compare_groups_tool",
        tool_output={
            "group_count": 3,
            "highest_group": "Engineering",
            "highest_value": 95000,
            "results": [
                {"group": "Engineering", "value": 95000},
                {"group": "Marketing", "value": 78000},
            ],
        },
    )

    assert evidence.tool_name == "compare_groups_tool"
    assert evidence.tool_output["highest_group"] == "Engineering"
    assert len(evidence.tool_output["results"]) == 2


def test_factual_claim_rejects_empty_statement():
    with pytest.raises(ValidationError):
        FactualClaim(
            claim_id="claim_3",
            statement="",
            claim_type="numerical_result",
            claim_data={"value": 42},
            evidence_ids=["evidence_1"],
        )


def test_factual_claim_rejects_empty_evidence_ids():
    with pytest.raises(ValidationError):
        FactualClaim(
            claim_id="claim_4",
            statement="Revenue increased.",
            claim_type="numerical_result",
            claim_data={"value": 42},
            evidence_ids=[],
        )


def test_factual_claim_rejects_invalid_claim_type():
    with pytest.raises(ValidationError):
        FactualClaim(
            claim_id="claim_5",
            statement="Some result was found.",
            claim_type="unsupported_type",
            claim_data={"value": 42},
            evidence_ids=["evidence_1"],
        )
