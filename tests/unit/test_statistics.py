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
