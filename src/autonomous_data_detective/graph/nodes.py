import json

from autonomous_data_detective.agents.schemas import EvidenceRecord
from autonomous_data_detective.core.config import get_settings
from autonomous_data_detective.graph.state import InvestigationState
from autonomous_data_detective.agents.schemas import (
    EvidenceRecord,
    FactualClaim,
    InvestigationReport,
)
from autonomous_data_detective.agents.evidence_validator import validate_claim

from autonomous_data_detective.agents.profiler import (
    build_profiler_agent,
)
from autonomous_data_detective.agents.investigator import (
    build_investigator_agent,
)

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


def find_evidence_for_claim(
    claim: FactualClaim,
    evidence_records: list[EvidenceRecord],
) -> list[EvidenceRecord]:
    """Find evidence matching a claim's structured metadata."""

    claim_data = claim.claim_data
    matching_records = []

    for evidence in evidence_records:
        output = evidence.tool_output

        if claim.claim_type == "group_comparison":
            if evidence.tool_name != "compare_groups_tool":
                continue

            if output.get("group_by") != claim_data.get("group_column"):
                continue

            if output.get("metric") != claim_data.get("metric_column"):
                continue

            if output.get("aggregation") != claim_data.get("aggregation"):
                continue

            matching_records.append(evidence)

        elif claim.claim_type == "numerical_result":
            if evidence.tool_name != "get_column_statistics_tool":
                continue

            expected_column = claim_data.get("column_name")
            result_key = claim_data.get("result_key", "value")

            if (
                expected_column is not None
                and output.get("column_name") != expected_column
            ):
                continue

            statistics = output.get("statistics")

            has_result = (
                isinstance(statistics, dict)
                and result_key in statistics
            ) or result_key in output

            if has_result:
                matching_records.append(evidence)

        elif claim.claim_type == "correlation":
            if evidence.tool_name != "analyze_correlation_tool":
                continue

            if output.get("column_a") != claim_data.get("column_a"):
                continue

            if output.get("column_b") != claim_data.get("column_b"):
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

            for claim in structured_report.claims:
                matching_evidence = find_evidence_for_claim(
                    claim,
                    evidence_records,
                )

                if not matching_evidence:
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

                # Use the IDs of the actual matched records, not IDs
                # supplied by the model.
                linked_claim = claim.model_copy(
                    update={
                        "evidence_ids": [
                            evidence.evidence_id
                            for evidence in matching_evidence
                        ]
                    }
                )

                validation_results.append(
                    validate_claim(linked_claim, matching_evidence)
                )

            validated_report = structured_report.model_copy(
                update={"validation_results": validation_results}
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
