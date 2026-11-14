
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from autonomous_data_detective.graph.question_validation import (
    QuestionInterpretation,
    interpret_question,
    validate_interpretation,
    validate_question,
)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "department": ["Sales", "IT", "Sales", "HR"],
            "salary": [50000, 70000, 60000, 55000],
            "revenue": [1000, 2000, 1500, 1200],
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


def make_interpretation(**overrides) -> QuestionInterpretation:
    values = {
        "operation": "aggregation",
        "referenced_columns": ["salary"],
        "metric_columns": ["salary"],
        "grouping_columns": [],
        "aggregation": "mean",
        "ambiguous": False,
        "clarification_question": None,
    }
    values.update(overrides)
    return QuestionInterpretation(**values)


def test_allows_aggregation_on_numeric_column(sample_csv: Path) -> None:
    result = validate_interpretation(
        make_interpretation(),
        str(sample_csv),
    )

    assert result["status"] == "valid"


def test_rejects_missing_metric_column(sample_csv: Path) -> None:
    result = validate_interpretation(
        make_interpretation(
            referenced_columns=["bonus"],
            metric_columns=["bonus"],
        ),
        str(sample_csv),
    )

    assert result["status"] == "invalid"
    assert "bonus" in result["message"].lower()


def test_rejects_average_of_categorical_column(
    sample_csv: Path,
) -> None:
    result = validate_interpretation(
        make_interpretation(
            referenced_columns=["department"],
            metric_columns=["department"],
        ),
        str(sample_csv),
    )

    assert result["status"] == "invalid"
    assert "numeric" in result["message"].lower()


def test_rejects_missing_grouping_column(sample_csv: Path) -> None:
    result = validate_interpretation(
        make_interpretation(
            operation="group_aggregation",
            referenced_columns=["region", "salary"],
            metric_columns=["salary"],
            grouping_columns=["region"],
        ),
        str(sample_csv),
    )

    assert result["status"] == "invalid"
    assert "region" in result["message"].lower()


def test_requests_clarification_for_ambiguous_question(
    sample_csv: Path,
) -> None:
    result = validate_interpretation(
        make_interpretation(
            operation="other",
            referenced_columns=["department"],
            metric_columns=[],
            aggregation="none",
            ambiguous=True,
            clarification_question=(
                "What criterion should determine the best department?"
            ),
        ),
        str(sample_csv),
    )

    assert result["status"] == "clarify"


def test_resolves_column_names_case_insensitively(
    sample_csv: Path,
) -> None:
    result = validate_interpretation(
        make_interpretation(
            referenced_columns=["SALARY"],
            metric_columns=["SALARY"],
        ),
        str(sample_csv),
    )

    assert result["status"] == "valid"


def test_allows_categorical_grouping_with_numeric_aggregation(
    sample_csv: Path,
) -> None:
    result = validate_interpretation(
        make_interpretation(
            operation="group_aggregation",
            referenced_columns=["department", "salary"],
            metric_columns=["salary"],
            grouping_columns=["department"],
        ),
        str(sample_csv),
    )

    assert result["status"] == "valid"


def test_interpreter_returns_structured_interpretation() -> None:
    expected = make_interpretation(
        operation="group_aggregation",
        referenced_columns=["region", "salary"],
        metric_columns=["salary"],
        grouping_columns=["region"],
    )

    with patch(
        "autonomous_data_detective.graph.question_validation."
        "ChatOpenAI"
    ) as mock_model:
        mock_model.return_value.with_structured_output.return_value.invoke.return_value = (
            expected
        )

        result = interpret_question(
            "Which region has the highest average salary?"
        )

    assert result == expected


def test_validate_question_handles_interpreter_failure(
    sample_csv: Path,
) -> None:
    with patch(
        "autonomous_data_detective.graph.question_validation."
        "interpret_question",
        side_effect=RuntimeError("API unavailable"),
    ):
        result = validate_question(
            "What is the average salary?",
            str(sample_csv),
        )

    assert result["status"] == "error"
    assert result["error_type"] == "RuntimeError"
