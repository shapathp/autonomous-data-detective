from typing import Any

from autonomous_data_detective.agents.schemas import (
    EvidenceRecord,
    FactualClaim,
)


def validate_claim(
    claim: FactualClaim,
    evidence_records: list[EvidenceRecord],
) -> dict[str, Any]:
    """Validate a factual claim against its referenced tool evidence."""

    evidence_by_id = {
        evidence.evidence_id: evidence
        for evidence in evidence_records
    }

    missing_evidence_ids = [
        evidence_id
        for evidence_id in claim.evidence_ids
        if evidence_id not in evidence_by_id
    ]

    if missing_evidence_ids:
        return {
            "claim_id": claim.claim_id,
            "status": "unverifiable",
            "reason": (
                "Referenced evidence records were not provided: "
                + ", ".join(missing_evidence_ids)
            ),
        }

    referenced_evidence = [
        evidence_by_id[evidence_id]
        for evidence_id in claim.evidence_ids
    ]

    if claim.claim_type == "group_comparison":
        return _validate_group_comparison(claim, referenced_evidence)

    if claim.claim_type == "numerical_result":
        return _validate_numerical_result(claim, referenced_evidence)

    if claim.claim_type == "correlation":
        return _validate_correlation(claim, referenced_evidence)

    return {
        "claim_id": claim.claim_id,
        "status": "unverifiable",
        "reason": (
            f"Automatic validation is not yet implemented for "
            f"claim type '{claim.claim_type}'."
        ),
    }



def _validate_group_comparison(
    claim: FactualClaim,
    evidence_records: list[EvidenceRecord],
) -> dict[str, Any]:
    """Check group-comparison claims against relevant tool evidence."""

    claim_data = claim.claim_data
    expected_group = claim_data.get("group")
    expected_value = claim_data.get("value")

    expected_group_column = claim_data.get("group_column")
    expected_metric_column = claim_data.get("metric_column")
    expected_aggregation = claim_data.get("aggregation")

    if (
        not expected_group_column
        or not expected_metric_column
        or not expected_aggregation
        or expected_group is None
        or expected_value is None
    ):
        return {
            "claim_id": claim.claim_id,
            "status": "unverifiable",
            "reason": (
                "The group-comparison claim must specify "
                "the group column, metric column, aggregation, "
                "expected group, and expected value."
            ),
        }

    relevant_evidence_found = False

    for evidence in evidence_records:
        output = evidence.tool_output

        # Confirm that the evidence describes the requested comparison.
        if (
            expected_group_column is not None
            and output.get("group_by") != expected_group_column
        ):
            continue

        if output.get("metric") != expected_metric_column:
            continue

        if (
            expected_aggregation is not None
            and output.get("aggregation") != expected_aggregation
        ):
            continue

        relevant_evidence_found = True

        # The tool sorts results in descending order, so its first
        # result is the highest group for the selected aggregation.
        results = output.get("results")

        if not isinstance(results, list) or not results:
            continue

        
        valid_results = [
            result
            for result in results
            if (
                isinstance(result, dict)
                and isinstance(result.get("value"), (int, float))
                and not isinstance(result.get("value"), bool)
            )
        ]

        if not valid_results:
            continue

        highest_result = max(
            valid_results,
            key=lambda result: result["value"],
        )

        highest_group = highest_result.get("group")
        highest_value = highest_result.get("value")


        # Only verify a highest-group claim when the requested group
        # and value match the actual top result.
        if (
            expected_group == highest_group
            and _values_match(highest_value, expected_value)
        ):
            return {
                "claim_id": claim.claim_id,
                "status": "verified",
                "reason": (
                    "The claimed group and value match the highest "
                    "result for the specified comparison."
                ),
                "evidence_ids": [evidence.evidence_id],
            }

        # A matching non-highest group/value pair is supported, but
        # does not establish that the group is the highest.
        for result in results:
            if (
                isinstance(result, dict)
                and result.get("group") == expected_group
                and _values_match(result.get("value"), expected_value)
            ):
                return {
                    "claim_id": claim.claim_id,
                    "status": "supported",
                    "reason": (
                        "The group and value match a tool result, but "
                        "the group is not the highest result."
                    ),
                    "evidence_ids": [evidence.evidence_id],
                }

    if not relevant_evidence_found:
        return {
            "claim_id": claim.claim_id,
            "status": "unverifiable",
            "reason": (
                "No referenced evidence matches the group column, "
                "metric, and aggregation specified by the claim."
            ),
        }

    return {
        "claim_id": claim.claim_id,
        "status": "unverifiable",
        "reason": (
            "Relevant evidence was found, but it did not establish "
            "the claimed group and value."
        ),
    }





def _validate_numerical_result(
    claim: FactualClaim,
    evidence_records: list[EvidenceRecord],
) -> dict[str, Any]:
    """Check numerical claims against relevant structured tool evidence."""

    claim_data = claim.claim_data
    expected_value = claim_data.get("value")
    target_key = claim_data.get("result_key", "value")

    relevant_evidence_found = False
    conflicting_value_found = False

    for evidence in evidence_records:
        output = evidence.tool_output

        expected_column = claim_data.get("column_name")
        actual_column = output.get("column_name")

        if expected_column is not None and actual_column != expected_column:
            continue

        expected_aggregation = claim_data.get("aggregation")
        actual_aggregation = output.get("aggregation")

        if (
            expected_aggregation is not None
            and actual_aggregation != expected_aggregation
        ):
            continue

        relevant_evidence_found = True

        # Statistics tools nest numerical values under "statistics".
        statistics = output.get("statistics")
        if isinstance(statistics, dict) and target_key in statistics:
            actual_value = statistics[target_key]
        else:
            actual_value = output.get(target_key)

        # A missing or null result cannot verify or contradict a claim.
        if actual_value is None:
            continue

        if _values_match(actual_value, expected_value):
            return {
                "claim_id": claim.claim_id,
                "status": "verified",
                "reason": (
                    f"The claimed value matches the tool output "
                    f"field '{target_key}'."
                ),
                "evidence_ids": [evidence.evidence_id],
            }

        conflicting_value_found = True

    if not relevant_evidence_found:
        return {
            "claim_id": claim.claim_id,
            "status": "unverifiable",
            "reason": (
                "No referenced evidence matches the column and "
                "aggregation specified by the claim."
            ),
        }

    return {
        "claim_id": claim.claim_id,
        "status": "unverifiable",
        "reason": (
            "Relevant evidence was found, but it did not establish "
            "the claimed numerical value. "
            + (
                "A different value was returned."
                if conflicting_value_found
                else "The requested result was missing or null."
            )
        ),
    }



def _validate_correlation(
    claim: FactualClaim,
    evidence_records: list[EvidenceRecord],
) -> dict[str, Any]:
    """Validate correlation claims against correlation tool evidence."""

    claim_data = claim.claim_data
    expected_value = claim_data.get("correlation")
    expected_column_a = claim_data.get("column_a")
    expected_column_b = claim_data.get("column_b")
    expected_method = claim_data.get("method", "pearson")

    if (
        not expected_column_a
        or not expected_column_b
        or expected_value is None
    ):
        return {
            "claim_id": claim.claim_id,
            "status": "unverifiable",
            "reason": (
                "The correlation claim must specify both columns "
                "and the expected correlation coefficient."
            ),
        }

    for evidence in evidence_records:
        if evidence.tool_name != "analyze_correlation_tool":
            continue

        output = evidence.tool_output

        if (
            expected_column_a is not None
            and output.get("column_a") != expected_column_a
        ):
            continue

        if (
            expected_column_b is not None
            and output.get("column_b") != expected_column_b
        ):
            continue

        if output.get("method") != expected_method:
            continue

        actual_value = output.get("correlation")

        if actual_value is None or expected_value is None:
            continue

        if _values_match(actual_value, expected_value):
            return {
                "claim_id": claim.claim_id,
                "status": "verified",
                "reason": (
                    "The claimed correlation coefficient matches "
                    "the correlation tool output for the specified "
                    "columns and method."
                ),
                "evidence_ids": [evidence.evidence_id],
            }

    return {
        "claim_id": claim.claim_id,
        "status": "unverifiable",
        "reason": (
            "No matching correlation evidence established the "
            "claimed coefficient for the specified columns and method."
        ),
    }

def _values_match(actual: Any, expected: Any) -> bool:
    """Compare values without treating booleans as numbers."""

    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual == expected

    if isinstance(actual, (int, float)) and isinstance(
        expected, (int, float)
    ):
        return abs(actual - expected) <= 1e-9

    return actual == expected
