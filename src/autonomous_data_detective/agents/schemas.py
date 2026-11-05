from pydantic import BaseModel, Field
from typing import Any, Literal


class EvidenceRecord(BaseModel):
    """Structured output returned by an analysis tool."""

    evidence_id: str = Field(
        min_length=1,
        description="Unique identifier for this evidence record.",
    )

    tool_name: str = Field(
        min_length=1,
        description="Name of the tool that produced the evidence.",
    )

    tool_output: dict[str, Any] = Field(
        description="Structured result returned by the analysis tool.",
    )


class FactualClaim(BaseModel):
    """A factual claim that can be checked against analysis evidence."""

    claim_id: str = Field(
        min_length=1,
        description="Unique identifier for this claim.",
    )

    statement: str = Field(
        min_length=1,
        description="Human-readable statement being evaluated.",
    )

    claim_type: Literal[
        "numerical_result",
        "group_comparison",
        "missing_data",
        "correlation",
        "interpretation",
        "other",
    ] = Field(
        description="Category used to determine how the claim should be checked.",
    )

    claim_data: dict[str, Any] = Field(
        description=(
            "Structured facts asserted by the claim, such as a value, "
            "column, group, or correlation coefficient."
        ),
    )

    evidence_ids: list[str] = Field(
        min_length=1,
        description="Identifiers of evidence records supporting this claim.",
    )

class InvestigationReport(BaseModel):
    """Structured report produced by the Investigator Agent."""

    summary: str = Field(
        min_length=1,
        description="A concise answer to the user's question.",
    )

    findings: list[str] = Field(
        min_length=1,
        description="Specific findings discovered during analysis.",
    )

    evidence: list[str] = Field(
        min_length=1,
        description="Tool results or observations supporting the findings.",
    )

    limitations: list[str] = Field(
        description=(
            "Relevant caveats, missing data, excluded rows, "
            "or constraints affecting the conclusions."
        )
    )

    recommendations: list[str] = Field(
        description="Practical next steps suggested by the analysis."
    )

    claims: list[FactualClaim] = Field(
        default_factory=list,
        description=(
            "Structured factual claims from the report, including "
            "the claim type, asserted values, and supporting evidence IDs."
        ),
    )

    validation_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Results of checking factual claims against tool evidence. "
            "Each result includes a claim ID, status, and reason."
        ),
    )
