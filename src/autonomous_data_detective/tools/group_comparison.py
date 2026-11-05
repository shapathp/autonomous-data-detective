from pathlib import Path
from typing import Any

import pandas as pd
from langchain.tools import tool


def compare_groups(
    csv_path: str,
    group_by: str,
    metric: str,
    aggregation: str = "mean",
) -> dict[str, Any]:
    """Compare a numeric metric across groups in a CSV dataset."""

    path = Path(csv_path)

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path.name}")

    
    df = pd.read_csv(path)

    column_lookup = {
        column.casefold(): column
        for column in df.columns
    }

    actual_group_by = column_lookup.get(group_by.casefold())
    actual_metric = column_lookup.get(metric.casefold())

    if actual_group_by is None:
        raise ValueError(f"Grouping column not found: {group_by}")

    if actual_metric is None:
        raise ValueError(f"Metric column not found: {metric}")

    group_by = actual_group_by
    metric = actual_metric


    if group_by == metric:
        raise ValueError("Grouping column and metric must be different.")

    allowed_aggregations = {"mean", "sum", "min", "max", "count"}

    if aggregation not in allowed_aggregations:
        raise ValueError(
            f"Unsupported aggregation: {aggregation}. "
            f"Choose from {sorted(allowed_aggregations)}."
        )

    if aggregation != "count" and not pd.api.types.is_numeric_dtype(
        df[metric]
    ):
        raise ValueError(
            f"Metric column '{metric}' must be numeric "
            f"for {aggregation} aggregation."
        )

    valid_df = df.dropna(subset=[group_by, metric])

    if valid_df.empty:
        raise ValueError(
            "No rows have both a group value and a metric value."
        )

    grouped = valid_df.groupby(group_by)[metric]

    if aggregation == "mean":
        comparison = grouped.mean()
    elif aggregation == "sum":
        comparison = grouped.sum()
    elif aggregation == "min":
        comparison = grouped.min()
    elif aggregation == "max":
        comparison = grouped.max()
    else:
        comparison = grouped.count()

    results = [
        {
            "group": str(group),
            "value": float(value),
        }
        for group, value in comparison.items()
    ]

    results.sort(key=lambda item: item["value"], reverse=True)

    return {
        "dataset_name": path.name,
        "group_by": group_by,
        "metric": metric,
        "aggregation": aggregation,
        "excluded_rows": int(len(df) - len(valid_df)),
        "group_count": len(results),
        "results": results,
        "highest_group": results[0]["group"],
        "highest_value": results[0]["value"],
    }


@tool
def compare_groups_tool(
    csv_path: str,
    group_by: str,
    metric: str,
    aggregation: str = "mean",
) -> dict[str, Any]:
    """Compare a numeric metric across categories in a CSV dataset.

    Use this tool to compare averages, totals, minimums, maximums,
    or non-missing record counts across groups.

    Args:
        csv_path: Path to the CSV dataset.
        group_by: Column defining groups, such as region or product.
        metric: Column to measure, such as revenue or quantity.
        aggregation: One of mean, sum, min, max, or count.
    """
    return compare_groups(csv_path, group_by, metric, aggregation)
