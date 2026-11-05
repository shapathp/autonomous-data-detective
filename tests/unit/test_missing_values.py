import pytest

from autonomous_data_detective.tools.missing_values import (
    analyze_missing_values,
    analyze_missing_values_tool,
)


CSV_PATH = "data/samples/sales.csv"


def test_missing_value_summary():
    result = analyze_missing_values(CSV_PATH)

    assert result["row_count"] == 10
    assert result["column_count"] == 5
    assert result["total_cells"] == 50
    assert result["total_missing_values"] == 2
    assert result["missing_cell_percentage"] == 4.0
    assert result["columns_with_missing_values"] == 2


def test_missing_columns_are_identified():
    result = analyze_missing_values(CSV_PATH)

    columns = {
        item["column"]: item
        for item in result["columns"]
    }

    assert columns["revenue"]["missing_count"] == 1
    assert columns["quantity"]["missing_count"] == 1
    assert columns["region"]["missing_count"] == 0


def test_missing_value_recommendations():
    result = analyze_missing_values(CSV_PATH)

    columns = {
        item["column"]: item
        for item in result["columns"]
    }

    assert "imputation" in columns["revenue"]["recommendation"]
    assert (
        columns["region"]["recommendation"]
        == "No missing values detected."
    )


def test_non_csv_file_is_rejected(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("example")

    with pytest.raises(ValueError, match="Only CSV"):
        analyze_missing_values(str(file_path))


def test_missing_file_is_rejected():
    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        analyze_missing_values("data/samples/not_found.csv")


def test_langchain_tool_can_be_invoked():
    result = analyze_missing_values_tool.invoke(
        {"csv_path": CSV_PATH}
    )

    assert result["total_missing_values"] == 2
    assert result["columns_with_missing_values"] == 2

def test_empty_dataset_is_rejected(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("revenue,quantity\n")

    with pytest.raises(ValueError, match="The dataset is empty"):
        analyze_missing_values(str(csv_file))

def test_dataset_with_all_missing_values(tmp_path):
    csv_file = tmp_path / "all_missing.csv"
    csv_file.write_text("revenue,quantity\n,\n,\n")

    result = analyze_missing_values(str(csv_file))

    assert result["row_count"] == 2
    assert result["column_count"] == 2
    assert result["total_cells"] == 4
    assert result["total_missing_values"] == 4
    assert result["missing_cell_percentage"] == 100.0
    assert result["columns_with_missing_values"] == 2