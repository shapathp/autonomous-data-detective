
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from autonomous_data_detective.core.config import get_settings


class QuestionInterpretation(BaseModel):
    """Structured interpretation of a user's data question."""

    operation: Literal[
        "profile",
        "aggregation",
        "group_aggregation",
        "comparison",
        "correlation",
        "missing_data",
        "other",
    ]

    referenced_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Every dataset column explicitly or semantically referenced "
            "by the question, including columns that might not exist."
        ),
    )

    metric_columns: list[str] = Field(default_factory=list)
    grouping_columns: list[str] = Field(default_factory=list)

    aggregation: Literal[
        "mean",
        "median",
        "mode",
        "sum",
        "count",
        "min",
        "max",
        "std",
        "variance",
        "none",
    ] = "none"

    ambiguous: bool = False
    clarification_question: str | None = None


def interpret_question(question: str) -> QuestionInterpretation:
    """Interpret a question without assuming that its columns exist."""

    settings = get_settings()

    model = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0,
        timeout=30,
        max_retries=1,
    )

    interpreter = model.with_structured_output(QuestionInterpretation)

    result = interpreter.invoke(
        [
            (
                "system",
                """
You interpret questions about arbitrary CSV datasets.

Identify the user's intended operation, metric columns, grouping columns,
aggregation, and ambiguity.

Rules:
- Do not assume any column exists.
- List every column the question refers to in referenced_columns,
  including requested columns that may be absent from the dataset.
- Preserve the column names requested by the user. Do not silently
  replace a missing column with a similar existing column.
- A grouping column is distinct from a metric column.
- For "average salary by region", use salary as the metric and
  region as the grouping column.
- For "average department", department is the metric column.
- Mean, median, sum, standard deviation, and variance are numerical
  aggregations. Mode and count can also apply to categorical data.
- "Best", "worst", or similar terms are ambiguous when the user has
  not specified a meaningful criterion.
- Do not answer the user's question or calculate any results.
- Return only the structured interpretation.
""".strip(),
            ),
            ("human", question),
        ]
    )

    return result


def validate_interpretation(
    interpretation: QuestionInterpretation,
    csv_path: str,
) -> dict[str, Any]:
    """Validate an interpreted question against the actual CSV schema."""

    path = Path(csv_path)

    if not path.is_file():
        return {
            "status": "invalid",
            "message": "The dataset file could not be found.",
        }

    try:
        df = pd.read_csv(path)
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        pd.errors.ParserError,
    ) :
        return {
            "status": "invalid",
            "message": "The dataset could not be read as a CSV file.",
        }

    if df.empty:
        return {
            "status": "invalid",
            "message": "The dataset contains no data rows.",
        }

    actual_columns = list(df.columns)

    def resolve_column(name: str) -> str | None:
        """Resolve a column name without case/whitespace sensitivity."""
        matches = [
            column
            for column in actual_columns
            if str(column).strip().casefold() == name.strip().casefold()
        ]
        return str(matches[0]) if len(matches) == 1 else None

    referenced = list(
        dict.fromkeys(
            interpretation.referenced_columns
            + interpretation.metric_columns
            + interpretation.grouping_columns
        )
    )

    missing_columns = [
        name for name in referenced if resolve_column(name) is None
    ]

    if missing_columns:
        return {
            "status": "invalid",
            "message": (
                f"Column(s) not found: {', '.join(missing_columns)}. "
                f"Available columns: {', '.join(map(str, actual_columns))}."
            ),
        }

    if interpretation.ambiguous:
        return {
            "status": "clarify",
            "message": (
                interpretation.clarification_question
                or "Please clarify what you want to measure or compare."
            ),
        }

    numeric_aggregations = {
        "mean",
        "median",
        "sum",
        "std",
        "variance",
    }

    if interpretation.aggregation in numeric_aggregations:
        for name in interpretation.metric_columns:
            column = resolve_column(name)

            if column is not None and not pd.api.types.is_numeric_dtype(
                df[column]
            ):
                return {
                    "status": "invalid",
                    "message": (
                        f"Cannot calculate {interpretation.aggregation} "
                        f"for '{column}' because it is not a numeric "
                        "column. You can group by categorical columns, "
                        "but this aggregation requires a numeric metric."
                    ),
                }

    if (
        interpretation.operation == "group_aggregation"
        and not interpretation.grouping_columns
    ):
        return {
            "status": "clarify",
            "message": (
                "I couldn't identify a grouping column. "
                "Which column should the results be grouped by?"
            ),
        }

    return {
        "status": "valid",
        "message": "The question passed dataset schema validation.",
    }


def validate_question(
    question: str,
    csv_path: str,
) -> dict[str, Any]:
    """Interpret and validate a natural-language dataset question."""

    try:
        interpretation = interpret_question(question)
    except Exception as exc: # noqa: BLE001 #This tell ruff The broad catch is deliberate here, rather than a general pattern to use everywhere
        return {
            "status": "error",
            "message": (
                "I couldn't interpret the question reliably. "
                "Please try rephrasing it."
            ),
            "error_type": type(exc).__name__,
        }

    return validate_interpretation(interpretation, csv_path)
