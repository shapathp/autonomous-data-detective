import json
import math

from autonomous_data_detective.agents.evidence_validator import validate_claim
from autonomous_data_detective.agents.investigator import (
    build_investigator_agent,
)
from autonomous_data_detective.agents.profiler import (
    build_profiler_agent,
)
from autonomous_data_detective.agents.schemas import (
    EvidenceRecord,
    FactualClaim,
    InvestigationReport,
)
from autonomous_data_detective.core.config import get_settings
from autonomous_data_detective.graph.state import InvestigationState


def extract_tool_evidence(messages: list) -> list[EvidenceRecord]:
    """Extract structured evidence from successful tool messages."""

    evidence_records = []
    evidence_number = 1

    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue

        tool_name = getattr(message, "name", None)
        content = getattr(message, "content", None)

        if not tool_name or not isinstance(content, str):
            continue

        try:
            tool_output = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            # Skip tool messages that do not contain valid JSON.
            continue

        if not isinstance(tool_output, dict):
            # Evidence must be a structured dictionary.
            continue

        evidence_records.append(
            EvidenceRecord(
                evidence_id=f"evidence_{evidence_number}",
                tool_name=tool_name,
                tool_output=tool_output,
            )
        )

        evidence_number += 1

    return evidence_records


def _same_column_name(left: object, right: object) -> bool:
    """Compare column names ignoring case and surrounding whitespace."""

    return (
        isinstance(left, str)
        and isinstance(right, str)
        and left.strip().casefold() == right.strip().casefold()
    )




def _get_highest_groups(tool_output: dict) -> tuple[list, float | None]:
    """Return the highest groups and their maximum finite numeric value."""

    highest_groups = tool_output.get("highest_groups") or []
    results = tool_output.get("results") or []

    numeric_results = [
        item
        for item in results
        if isinstance(item, dict)
        and isinstance(item.get("value"), (int, float))
        and not isinstance(item.get("value"), bool)
        and math.isfinite(item["value"])
        and item.get("group") is not None
    ]

    if not numeric_results:
        return highest_groups, None

    maximum = max(item["value"] for item in numeric_results)

    derived_groups = list(
        dict.fromkeys(
            item["group"]
            for item in numeric_results
            if item["value"] == maximum
        )
    )

    return derived_groups, maximum

def find_evidence_for_claim(
    claim: FactualClaim,
    evidence_records: list[EvidenceRecord],
) -> list[EvidenceRecord]:
    """Find tool evidence matching a claim's structured metadata."""

    claim_data = claim.claim_data
    matching_records = []

    for evidence in evidence_records:
        output = evidence.tool_output

        if claim.claim_type == "group_comparison":
            if evidence.tool_name != "compare_groups_tool":
                continue

            if not _same_column_name(
                output.get("group_by"),
                claim_data.get("group_column"),
            ):
                continue

            if not _same_column_name(
                output.get("metric"),
                claim_data.get("metric_column"),
            ):
                continue

            if output.get("aggregation") != claim_data.get("aggregation"):
                continue

            matching_records.append(evidence)


        elif claim.claim_type == "numerical_result":
            if evidence.tool_name != "get_column_statistics_tool":
                continue

            # Support both existing and newer claim metadata conventions.
            expected_column = (
                claim_data.get("column_name")
                or claim_data.get("metric_column")
                or claim_data.get("column")
            )

            if not isinstance(expected_column, str) or not expected_column.strip():
                continue

            if not _same_column_name(
                output.get("column_name"),
                expected_column,
            ):
                continue

            statistics = output.get("statistics")

            # An explicit result_key takes precedence over aggregation.
            result_key = claim_data.get("result_key")

            if result_key is not None and (
                not isinstance(result_key, str)
                or not result_key.strip()
            ):
                continue

            if result_key is not None:
                candidate_keys = [result_key]
            else:
                aggregation = (
                    claim_data.get("aggregation")
                    or claim_data.get("metric")
                )

                if aggregation is None:
                    candidate_keys = ["value"]
                else:
                    normalized_aggregation = (
                        str(aggregation).strip().lower().replace("-", "_")
                    )

                    aggregation_aliases = {
                        "average": ["average", "mean"],
                        "mean": ["mean", "average"],
                        "standard deviation": [
                            "standard deviation",
                            "standard_deviation",
                            "std",
                            "std_dev",
                            "stddev",
                        ],
                        "standard_deviation": [
                            "standard_deviation",
                            "standard deviation",
                            "std",
                            "std_dev",
                            "stddev",
                        ],
                        "std": ["std", "standard_deviation", "std_dev"],
                        "std_dev": ["std_dev", "std", "standard_deviation"],
                        "stddev": ["stddev", "std", "standard_deviation"],
                        "variance": ["variance", "var"],
                        "var": ["var", "variance"],
                    }

                    candidate_keys = aggregation_aliases.get(
                        normalized_aggregation,
                        [normalized_aggregation],
                    )

            has_result = any(
                (
                    isinstance(statistics, dict)
                    and key in statistics
                )
                or key in output
                for key in candidate_keys
            )

            if has_result:
                matching_records.append(evidence)


        elif claim.claim_type == "correlation":
            if evidence.tool_name != "analyze_correlation_tool":
                continue

            if not _same_column_name(
                output.get("column_a"),
                claim_data.get("column_a"),
            ):
                continue

            if not _same_column_name(
                output.get("column_b"),
                claim_data.get("column_b"),
            ):
                continue

            if output.get("method") != claim_data.get(
                "method", "pearson"
            ):
                continue

            if output.get("correlation") is None:
                continue

            matching_records.append(evidence)

    return matching_records


def check_step_limit(state: InvestigationState) -> None:
    """Prevent agent execution after the configured step limit."""

    settings = get_settings()
    steps_taken = state.get("steps_taken", 0)

    if steps_taken >= settings.max_investigation_steps:
        raise RuntimeError(
            "Maximum investigation steps reached: "
            f"{settings.max_investigation_steps}"
        )


def profile_node(state: InvestigationState) -> dict:
    """Run the Profiler Agent and store its findings."""

    check_step_limit(state)

    try:
        agent = build_profiler_agent()

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Profile the CSV dataset at "
                            f"{state['csv_path']}. "
                            "Summarize its structure, missing values, "
                            "duplicate rows, and notable statistics."
                        ),
                    }
                ]
            }
        )

        final_message = result["messages"][-1]
        profile_result = final_message.content

        if not isinstance(profile_result, str):
            profile_result = str(profile_result)

        return {
            "profile_result": profile_result,
            "steps_taken": state.get("steps_taken", 0) + 1,
        }

    except Exception as exc:
        return {
            "errors": [
                *state.get("errors", []),
                f"Profiler failed: {type(exc).__name__}: {exc}",
            ],
            "steps_taken": state.get("steps_taken", 0) + 1,
        }


def investigate_node(state: InvestigationState) -> dict:
    """Run the Investigator Agent and validate its structured claims."""

    check_step_limit(state)

    try:
        agent = build_investigator_agent()

        profile_result = state.get("profile_result", "")

        if profile_result:
            profile_context = (
                "Profiler Agent findings:\n"
                f"{profile_result}\n\n"
                "Use these findings to guide your investigation. "
                "Consider any reported data-quality limitations, "
                "but verify numerical claims using the available "
                "analysis tools."
            )
        else:
            profile_context = (
                "No profiler findings are available. "
                "Use the available analysis tools to inspect "
                "the data as needed."
            )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{state['user_question']}\n\n"
                            f"Dataset path: {state['csv_path']}\n\n"
                            f"{profile_context}\n\n"
                            "Use the available analysis tools to answer "
                            "the question. Base numerical claims on "
                            "tool outputs and explain relevant limitations."
                        ),
                    }
                ]
            }
        )

        structured_report = result.get("structured_response")

        if isinstance(structured_report, InvestigationReport):
            messages = result.get("messages", [])
            evidence_records = extract_tool_evidence(messages)

            validation_results = []
            tie_statements = []
            linked_claims = []

            for claim in structured_report.claims:
                matching_evidence = find_evidence_for_claim(
                    claim,
                    evidence_records,
                )

                if not matching_evidence:
                    # Do not preserve evidence IDs supplied by the LLM
                    # when no matching tool evidence was found.
                    linked_claims.append(
                        claim.model_copy(update={"evidence_ids": []})
                    )

                    validation_results.append(
                        {
                            "claim_id": claim.claim_id,
                            "status": "unverifiable",
                            "reason": (
                                "No tool evidence matched the claim's "
                                "structured metadata."
                            ),
                            "evidence_ids": [],
                        }
                    )
                    continue

                # Deterministically detect tied winners using
                # the actual evidence matched to this claim.
                if claim.claim_type == "group_comparison":
                    for evidence in matching_evidence:
                        tool_output = getattr(
                            evidence,
                            "tool_output",
                            {},
                        )

                        if not isinstance(tool_output, dict):
                            continue


                        highest_groups, maximum = _get_highest_groups(tool_output)

                        # Only correct the report when numeric results establish a valid tie.
                        if maximum is None:
                            continue

                        highest_value = tool_output.get("highest_value")
                        if (
                            not isinstance(highest_value, (int, float))
                            or isinstance(highest_value, bool)
                            or not math.isfinite(highest_value)
                            or highest_value != maximum
                        ):
                            highest_value = maximum
                        # Fall back to the maximum derived from finite numeric results.
                        if highest_value is None:
                            highest_value = maximum

                        # A single winner requires no tie correction.
                        if len(highest_groups) < 2:
                            continue

                        # Fallback for evidence that provides tied groups but no value.
                        if highest_value is None:
                            results = tool_output.get("results") or []
                            highest_value = next(
                                (
                                    item.get("value")
                                    for item in results
                                    if isinstance(item, dict)
                                    and item.get("group") == highest_groups[0]
                                ),
                                None,
                            )


                        groups_text = ", ".join(
                            str(group)
                            for group in highest_groups
                        )

                        if highest_value is None:
                            statement = (
                                f"{groups_text} were tied for the "
                                "highest result."
                            )
                        else:
                            statement = (
                                f"{groups_text} were tied for the "
                                f"highest result ({highest_value})."
                            )

                        if statement not in tie_statements:
                            tie_statements.append(statement)

                # Use IDs from the actual matched tool evidence.
                linked_claim = claim.model_copy(
                    update={
                        "evidence_ids": [
                            evidence.evidence_id
                            for evidence in matching_evidence
                        ]
                    }
                )

                linked_claims.append(linked_claim)

                validation_results.append(
                    validate_claim(linked_claim, matching_evidence)
                )

            report_updates = {
                "claims": linked_claims,
                "validation_results": validation_results,
            }

            if tie_statements:
                # Evidence-backed tie information takes precedence
                # over an LLM-generated summary that implies a
                # unique winner.
                report_updates["summary"] = " ".join(tie_statements)

                corrected_findings = []

                for finding in structured_report.findings:
                    finding_text = str(finding).casefold()

                    # Remove a unique-winner finding only when it refers to
                    # a metric whose actual evidence establishes a tie.
                    unique_winner_phrases = (
                        "unique winner",
                        "uniquely highest",
                        "unique leader",
                        "sole winner",
                        "only winner",
                        "single winner",
                    )

                    claims_a_unique_winner = any(
                        phrase in finding_text
                        for phrase in unique_winner_phrases
                    )

                    if claims_a_unique_winner:
                        finding_refers_to_tied_metric = False

                        for claim in linked_claims:
                            if claim.claim_type != "group_comparison":
                                continue

                            claim_evidence = [
                                evidence
                                for evidence in evidence_records
                                if evidence.evidence_id in claim.evidence_ids
                            ]

                            for evidence in claim_evidence:
                                tool_output = evidence.tool_output

                                if not isinstance(tool_output, dict):
                                    continue

                                highest_groups, _ = _get_highest_groups(tool_output)

                                if len(highest_groups) < 2:
                                    continue

                                metric = str(
                                    tool_output.get("metric")
                                    or claim.claim_data.get("metric_column")
                                    or ""
                                ).casefold().replace("_", " ").strip()

                                if not metric:
                                    continue

                                # Match the metric name, including names such as
                                # "total_revenue" when the finding says "total revenue".
                                metric_tokens = [
                                    token for token in metric.split() if len(token) > 2
                                ]

                                mentions_metric = (
                                    metric in finding_text
                                    or (
                                        bool(metric_tokens)
                                        and all(
                                            token in finding_text
                                            for token in metric_tokens
                                        )
                                    )
                                )

                                if mentions_metric:
                                    finding_refers_to_tied_metric = True
                                    break

                            if finding_refers_to_tied_metric:
                                break

                        if finding_refers_to_tied_metric:
                            continue

                    # Remove a partial-tie finding only when it matches a
                    # claim whose own evidence establishes a tie.
                    finding_mentions_partial_tie = False

                    for claim in linked_claims:
                        if claim.claim_type != "group_comparison":
                            continue

                        claim_evidence = [
                            evidence
                            for evidence in evidence_records
                            if evidence.evidence_id in claim.evidence_ids
                        ]

                        for evidence in claim_evidence:
                            tool_output = evidence.tool_output

                            if not isinstance(tool_output, dict):
                                continue

                            highest_groups, _ = _get_highest_groups(tool_output)

                            if len(highest_groups) < 2:
                                continue

                            mentioned_groups = [
                                group
                                for group in highest_groups
                                if str(group).casefold() in finding_text
                            ]

                            if (
                                mentioned_groups
                                and len(mentioned_groups) < len(highest_groups)
                                and "highest" in finding_text
                            ):
                                # Only suppress the finding if it also refers to
                                # this claim's metric, not an unrelated metric.
                                metric = str(
                                    claim.claim_data.get("metric_column", "")
                                ).casefold()

                                if metric and metric in finding_text:
                                    finding_mentions_partial_tie = True
                                    break

                        if finding_mentions_partial_tie:
                            break

                    if finding_mentions_partial_tie:
                        continue
                    corrected_findings.append(finding)

                # Add the evidence-backed tie statements to findings.
                for statement in tie_statements:
                    if statement not in corrected_findings:
                        corrected_findings.append(statement)

                report_updates["findings"] = corrected_findings

            validated_report = structured_report.model_copy(
                update=report_updates
            )

            investigation_result = validated_report.model_dump_json(
                indent=2
            )

        elif structured_report is not None:
            investigation_result = str(structured_report)

        else:
            # Preserve compatibility with text-only agent responses
            # and existing tests that use mocked agents.
            final_message = result["messages"][-1]
            investigation_result = final_message.content

            if not isinstance(investigation_result, str):
                investigation_result = str(investigation_result)

        return {
            "investigation_result": investigation_result,
            "steps_taken": state.get("steps_taken", 0) + 1,
        }

    except Exception as exc:
        return {
            "errors": [
                *state.get("errors", []),
                f"Investigator failed: {type(exc).__name__}: {exc}",
            ],
            "steps_taken": state.get("steps_taken", 0) + 1,
        }
