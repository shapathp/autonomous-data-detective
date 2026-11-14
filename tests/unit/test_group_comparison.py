import pytest

from autonomous_data_detective.tools.group_comparison import (
    compare_groups,
    compare_groups_tool,
)

CSV_PATH = "data/samples/sales.csv"


def test_mean_revenue_by_region():
    result = compare_groups(
        CSV_PATH, "region", "revenue", "mean"
    )

    assert result["group_count"] == 4
    assert result["highest_group"] == "North"
    assert result["highest_value"] == 56000.0
    assert result["results"][0]["group"] == "North"
    assert result["excluded_rows"] == 1


def test_group_and_metric_column_names_are_case_insensitive():
    result = compare_groups(
        CSV_PATH, "Region", "Revenue", "mean"
    )

    assert result["highest_group"] == "North"
    assert result["highest_value"] == 56000.0
    assert result["group_by"] == "region"
    assert result["metric"] == "revenue"



def test_sum_revenue_by_product():
    result = compare_groups(
        CSV_PATH, "product", "revenue", "sum"
    )

    assert result["highest_group"] == "Laptop"
    assert result["highest_value"] == 225000.0


def test_missing_grouping_column_is_rejected():
    with pytest.raises(ValueError, match="Grouping column not found"):
        compare_groups(CSV_PATH, "department", "revenue")


def test_missing_metric_column_is_rejected():
    with pytest.raises(ValueError, match="Metric column not found"):
        compare_groups(CSV_PATH, "region", "profit")


def test_unsupported_aggregation_is_rejected():
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        compare_groups(CSV_PATH, "region", "revenue", "median")


def test_nonnumeric_metric_is_rejected_for_mean():
    with pytest.raises(ValueError, match="must be numeric"):
        compare_groups(CSV_PATH, "region", "product", "mean")


def test_langchain_tool_can_be_invoked():
    result = compare_groups_tool.invoke(
        {
            "csv_path": CSV_PATH,
            "group_by": "region",
            "metric": "revenue",
            "aggregation": "mean",
        }
    )

    assert result["highest_group"] == "North"
    assert result["group_count"] == 4

def test_boolean_metric_is_rejected_for_mean(tmp_path):
    import pandas as pd

    csv_file = tmp_path / "boolean_metrics.csv"
    pd.DataFrame({
        "department": ["Engineering", "Marketing"],
        "is_active": [True, False],
    }).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="must be numeric"):
        compare_groups(
            str(csv_file),
            "department",
            "is_active",
            "mean",
        )

def test_mean_excludes_non_finite_metric_values(tmp_path):

    csv_file = tmp_path / "non_finite_groups.csv"
    csv_file.write_text(
        "department,revenue\n"
        "Sales,10\n"
        "Sales,inf\n"
        "Sales,20\n"
    )

    result = compare_groups(
        str(csv_file),
        "department",
        "revenue",
        "mean",
    )

    assert result["results"] == [
        {"group": "Sales", "value": 15.0}
    ]
    assert result["excluded_rows"] == 1

def test_count_supports_categorical_metrics(tmp_path):
    import pandas as pd

    csv_file = tmp_path / "categorical_count.csv"
    pd.DataFrame({
        "department": ["Sales", "Sales", "Engineering"],
        "employee": ["Alice", "Bob", "Charlie"],
    }).to_csv(csv_file, index=False)

    result = compare_groups(
        str(csv_file),
        "department",
        "employee",
        "count",
    )

    results = {
        item["group"]: item["value"]
        for item in result["results"]
    }

    assert results == {"Sales": 2.0, "Engineering": 1.0}


def test_non_csv_file_is_rejected(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text(
        "department,revenue\nSales,100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Only CSV files are supported"):
        compare_groups(str(file_path), "department", "revenue")


def test_missing_csv_file_is_rejected(tmp_path):
    file_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        compare_groups(str(file_path), "department", "revenue")


def test_same_grouping_and_metric_column_is_rejected():
    with pytest.raises(
        ValueError,
        match="Grouping column and metric must be different",
    ):
        compare_groups(CSV_PATH, "revenue", "revenue")


def test_no_valid_rows_is_rejected(tmp_path):
    file_path = tmp_path / "no_valid_rows.csv"
    file_path.write_text(
        "department,revenue\nSales,\n,100\n,\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="No rows have both a group value and a metric value",
    ):
        compare_groups(str(file_path), "department", "revenue")



def test_min_aggregation(tmp_path):
    file_path = tmp_path / "min_values.csv"
    file_path.write_text(
        "department,revenue\n"
        "Sales,30\n"
        "Sales,10\n"
        "Engineering,20\n",
        encoding="utf-8",
    )

    result = compare_groups(
        str(file_path),
        "department",
        "revenue",
        "min",
    )

    assert result["aggregation"] == "min"
    assert result["results"] == [
        {"group": "Engineering", "value": 20.0},
        {"group": "Sales", "value": 10.0},
    ]
    assert result["highest_group"] == "Engineering"
    assert result["highest_value"] == 20.0


def test_max_aggregation(tmp_path):
    file_path = tmp_path / "max_values.csv"
    file_path.write_text(
        "department,revenue\n"
        "Sales,30\n"
        "Sales,10\n"
        "Engineering,20\n",
        encoding="utf-8",
    )

    result = compare_groups(
        str(file_path),
        "department",
        "revenue",
        "max",
    )

    assert result["aggregation"] == "max"
    assert result["results"] == [
        {"group": "Sales", "value": 30.0},
        {"group": "Engineering", "value": 20.0},
    ]
    assert result["highest_group"] == "Sales"
    assert result["highest_value"] == 30.0


def test_tied_highest_groups_are_all_returned(tmp_path):
    file_path = tmp_path / "tied_groups.csv"
    file_path.write_text(
        "department,salary\n"
        "Engineering,95000\n"
        "Marketing,95000\n"
        "Sales,70000\n",
        encoding="utf-8",
    )

    result = compare_groups(
        str(file_path),
        "department",
        "salary",
        "mean",
    )

    assert result["highest_value"] == 95000.0
    assert set(result["highest_groups"]) == {
        "Engineering",
        "Marketing",
    }
    assert len(result["highest_groups"]) == 2


def test_single_highest_group_returns_one_item_in_highest_groups(
    tmp_path,
):
    file_path = tmp_path / "unique_highest_group.csv"
    file_path.write_text(
        "department,salary\n"
        "Engineering,95000\n"
        "Marketing,85000\n"
        "Sales,70000\n",
        encoding="utf-8",
    )

    result = compare_groups(
        str(file_path),
        "department",
        "salary",
        "mean",
    )

    assert result["highest_group"] == "Engineering"
    assert result["highest_value"] == 95000.0
    assert result["highest_groups"] == ["Engineering"]


def test_mean_rejects_non_finite_aggregate(tmp_path):
    csv_file = tmp_path / "large_values.csv"
    csv_file.write_text(
        "department,revenue\n"
        "Sales,1.7e308\n"
        "Sales,1.7e308\n"
        "Engineering,100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite result"):
        compare_groups(
            str(csv_file),
            "department",
            "revenue",
            "mean",
        )