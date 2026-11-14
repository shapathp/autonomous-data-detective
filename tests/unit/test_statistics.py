import pandas as pd
import pytest

from autonomous_data_detective.tools.statistics import (
    get_column_statistics,
    get_column_statistics_tool,
)


def test_numeric_column_statistics(tmp_path):
    csv_file = tmp_path / "metrics.csv"
    pd.DataFrame({"revenue": [10, 20, 30, None]}).to_csv(
        csv_file, index=False
    )

    result = get_column_statistics(str(csv_file), "revenue")

    assert result["total_rows"] == 4
    assert result["missing_count"] == 1
    assert result["missing_percentage"] == 25.0
    assert result["statistics"]["mean"] == 20.0
    assert result["statistics"]["median"] == 20.0


def test_column_name_is_case_insensitive(tmp_path):
    csv_file = tmp_path / "metrics.csv"
    pd.DataFrame({"revenue": [10, 20, 30]}).to_csv(
        csv_file, index=False
    )

    result = get_column_statistics(
        str(csv_file), "Revenue"
    )

    assert result["column_name"] == "revenue"
    assert result["statistics"]["mean"] == 20.0



def test_categorical_column_statistics(tmp_path):
    csv_file = tmp_path / "regions.csv"
    pd.DataFrame(
        {"region": ["North", "South", "North", "East"]}
    ).to_csv(csv_file, index=False)

    result = get_column_statistics(str(csv_file), "region")

    assert result["unique_count"] == 3
    assert {"value": "North", "count": 2} in result["top_values"]


def test_unknown_column_is_rejected(tmp_path):
    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"revenue": [10, 20]}).to_csv(
        csv_file, index=False
    )

    with pytest.raises(ValueError, match="Column not found"):
        get_column_statistics(str(csv_file), "profit")


def test_statistics_tool_can_be_invoked():
    result = get_column_statistics_tool.invoke(
        {
            "csv_path": "data/samples/sales.csv",
            "column_name": "revenue",
        }
    )

    assert result["column_name"] == "revenue"
    assert result["missing_count"] == 1

def test_boolean_column_is_not_treated_as_numeric_statistics(tmp_path):
    csv_file = tmp_path / "flags.csv"
    pd.DataFrame({"is_active": [True, False, True]}).to_csv(
        csv_file, index=False
    )

    result = get_column_statistics(str(csv_file), "is_active")

    assert "statistics" not in result
    assert result["top_values"] == [
        {"value": "True", "count": 2},
        {"value": "False", "count": 1},
    ]

def test_non_finite_values_are_reported_and_excluded(tmp_path):
    csv_file = tmp_path / "non_finite.csv"
    csv_file.write_text("revenue\n10\ninf\n-inf\n20\n")

    result = get_column_statistics(str(csv_file), "revenue")

    assert result["statistics"]["mean"] == 15.0
    assert result["missing_count"] == 0
    assert result["non_finite_count"] == 2



def test_non_csv_file_is_rejected():
    with pytest.raises(
        ValueError,
        match="Only CSV files are supported",
    ):
        get_column_statistics("data.txt", "revenue")


def test_missing_csv_file_is_rejected(tmp_path):
    file_path = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match="CSV file not found",
    ):
        get_column_statistics(str(file_path), "revenue")


def test_get_column_statistics_includes_variance(tmp_path):
    csv_path = tmp_path / "numbers.csv"
    pd.DataFrame({"salary": [10, 20, 30]}).to_csv(
        csv_path,
        index=False,
    )

    result = get_column_statistics(
        str(csv_path),
        "salary",
    )

    assert result["statistics"]["mean"] == 20.0
    assert result["statistics"]["variance"] == 100.0
    assert result["statistics"]["std"] == 10.0

def test_statistics_are_none_when_all_numeric_values_are_non_finite(
    tmp_path,
):
    csv_file = tmp_path / "all_non_finite.csv"
    csv_file.write_text("revenue\ninf\n-inf\n")

    result = get_column_statistics(str(csv_file), "revenue")

    assert result["non_finite_count"] == 2
    assert result["missing_count"] == 0
    assert result["statistics"] is None