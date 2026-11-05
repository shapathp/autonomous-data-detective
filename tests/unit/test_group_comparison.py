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
