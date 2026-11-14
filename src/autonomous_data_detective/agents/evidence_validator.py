import math
from decimal import Decimal, InvalidOperation
from typing import Any

from autonomous_data_detective.agents.schemas import (
    EvidenceRecord,
    FactualClaim,
)


def _same_column_name(left: object, right: object) -> bool:
    """Compare column names ignoring case and surrounding whitespace."""

    return (
        isinstance(left, str)
        and isinstance(right, str)
        and left.strip().casefold() == right.strip().casefold()
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
            "Automatic validation is not yet implemented for "
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

        # Match the requested columns regardless of case or whitespace.
        if not _same_column_name(
            output.get("group_by"),
            expected_group_column,
        ):
            continue

        if not _same_column_name(
            output.get("metric"),
            expected_metric_column,
        ):
            continue

        if output.get("aggregation") != expected_aggregation:
            continue

        relevant_evidence_found = True

        # Determine the highest group from the actual tool results.
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
                and math.isfinite(result["value"])
                and result.get("group") is not None
            )
        ]

        if not valid_results:
            continue


        highest_value = max(
            result["value"]
            for result in valid_results
        )

        # A group is a highest group if its value equals the maximum.
        # This handles multiple groups tied at the maximum value.
        is_claimed_group_highest = any(
            result.get("group") == expected_group
            and _values_match(result["value"], highest_value)
            for result in valid_results
        )

        # Verify a highest-group claim if the claimed group is tied
        # for the maximum and the claimed value matches that maximum.
        if (
            is_claimed_group_highest
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

        # A matching non-highest group/value pair is supported,
        # but it does not establish that the group is the highest.
        for result in results:
            if not isinstance(result, dict):
                continue

            result_value = result.get("value")

            if (
                result.get("group") == expected_group
                and isinstance(result_value, (int, float))
                and not isinstance(result_value, bool)
                and _values_match(result_value, expected_value)
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
    """Validate numerical claims against structured tool evidence."""

    claim_data = claim.claim_data
    expected_value = claim_data.get("value")

    expected_column = (
        claim_data.get("column_name")
        or claim_data.get("metric_column")
        or claim_data.get("column")
    )

    expected_aggregation = (
        claim_data.get("aggregation")
        or claim_data.get("metric")
    )
    explicit_result_key = claim_data.get("result_key")

    if explicit_result_key is not None and (
        not isinstance(explicit_result_key, str)
        or not explicit_result_key.strip()
    ):
        return {
            "claim_id": claim.claim_id,
            "status": "unverifiable",
            "reason": "The requested result key must be a non-empty string.",
        }

    # Map common aggregation names to a canonical form.
    aggregation_aliases = {
        "mean": {"mean", "average"},
        "average": {"mean", "average"},
        "median": {"median"},
        "mode": {"mode"},
        "std": {
            "std",
            "std_dev",
            "stddev",
            "standard_deviation",
            "standard deviation",
        },
        "std_dev": {
            "std",
            "std_dev",
            "stddev",
            "standard_deviation",
            "standard deviation",
        },
        "stddev": {
            "std",
            "std_dev",
            "stddev",
            "standard_deviation",
            "standard deviation",
        },
        "standard_deviation": {
            "std",
            "std_dev",
            "stddev",
            "standard_deviation",
            "standard deviation",
        },
        "standard deviation": {
            "std",
            "std_dev",
            "stddev",
            "standard_deviation",
            "standard deviation",
        },
        "variance": {"variance", "var"},
        "var": {"variance", "var"},
        "sum": {"sum"},
        "min": {"min", "minimum"},
        "minimum": {"min", "minimum"},
        "max": {"max", "maximum"},
        "maximum": {"max", "maximum"},
        "count": {"count"},
    }

    def normalize_aggregation(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        return value.strip().casefold().replace("-", "_")

    def aggregation_keys(value: str) -> set[str]:
        normalized = normalize_aggregation(value)
        if normalized is None:
            return set()
        return aggregation_aliases.get(normalized, {normalized})

    # Determine which output keys can contain the requested result.
    if explicit_result_key is not None:
        candidate_keys = [explicit_result_key]
    elif expected_aggregation is not None:
        candidate_keys = list(aggregation_keys(expected_aggregation))
    else:
        candidate_keys = ["value"]

    relevant_evidence_found = False
    conflicting_value_found = False

    for evidence in evidence_records:
        output = evidence.tool_output

        actual_column = output.get("column_name")

        if not isinstance(expected_column, str) or not expected_column.strip():
            if isinstance(actual_column, str) and actual_column.strip():
                continue
        elif not _same_column_name(actual_column, expected_column):
            continue

        actual_aggregation = output.get("aggregation")

        # If the tool explicitly reports its aggregation, check that it
        # agrees with the claim. Otherwise, identify the statistic by key.
        if (
            expected_aggregation is not None
            and actual_aggregation is not None
            and not (
                aggregation_keys(expected_aggregation)
                & aggregation_keys(actual_aggregation)
            )
        ):
            continue


        statistics = output.get("statistics")
        matching_values = []

        for key in candidate_keys:
            if isinstance(statistics, dict) and key in statistics:
                matching_values.append((key, statistics[key]))
            elif key in output:
                matching_values.append((key, output[key]))

        if not matching_values:
            continue

        relevant_evidence_found = True

        # Conflicting aliases cannot independently establish one value.
        non_null_values = [
            (key, value)
            for key, value in matching_values
            if value is not None
        ]

        if not non_null_values:
            continue

        reference_value = non_null_values[0][1]

        if any(
            not _values_match(value, reference_value)
            for _, value in non_null_values[1:]
        ):
            conflicting_value_found = True
            continue

        matched_key, actual_value = non_null_values[0]


        relevant_evidence_found = True

        # Missing or null values cannot establish a numerical claim.
        if actual_value is None:
            continue

        if _values_match(actual_value, expected_value):
            return {
                "claim_id": claim.claim_id,
                "status": "verified",
                "reason": (
                    "The claimed value matches the tool output "
                    f"field '{matched_key}'."
                ),
                "evidence_ids": [evidence.evidence_id],
            }

        conflicting_value_found = True

    if not relevant_evidence_found:
        return {
            "claim_id": claim.claim_id,
            "status": "unverifiable",
            "reason": (
                "No referenced evidence matches the column, aggregation, "
                "and result field specified by the claim."
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
        or isinstance(expected_value, bool)
        or not isinstance(expected_value, (int, float))
        or not math.isfinite(expected_value)
        or not -1 <= expected_value <= 1
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

        if not _same_column_name(
            output.get("column_a"),
            expected_column_a,
        ):
            continue

        if not _same_column_name(
            output.get("column_b"),
            expected_column_b,
        ):
            continue

        if output.get("method") != expected_method:
            continue

        actual_value = output.get("correlation")

        if (
            isinstance(actual_value, bool)
            or not isinstance(actual_value, (int, float))
            or not math.isfinite(actual_value)
            or not -1 <= actual_value <= 1
        ):
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
    """Compare values while accounting for displayed numeric precision."""

    if isinstance(actual, bool) or isinstance(expected, bool):
        return (
            isinstance(actual, bool)
            and isinstance(expected, bool)
            and actual == expected
        )

    if isinstance(actual, (int, float)) and isinstance(
        expected, (int, float)
    ):
        if not math.isfinite(actual) or not math.isfinite(expected):
            return False

        try:
            expected_decimal = Decimal(str(expected))
            exponent = expected_decimal.as_tuple().exponent

            # Allow half a unit at the last displayed decimal place.
            # Integer claims remain strict apart from floating-point noise.
            rounding_tolerance = (
                float(Decimal("0.5") * (Decimal(10) ** exponent))
                if exponent < 0
                else 0.0
            )

            tolerance = max(
                1e-9,
                rounding_tolerance,
                4 * math.ulp(float(expected)),
            )

        except (InvalidOperation, ValueError, OverflowError):
            return actual == expected

        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=tolerance,
        )

    return actual == expected
