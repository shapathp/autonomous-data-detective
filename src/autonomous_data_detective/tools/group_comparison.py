import math
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

    # Resolve column names case-insensitively.
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

    if aggregation != "count" and (
        not pd.api.types.is_numeric_dtype(df[metric])
        or pd.api.types.is_bool_dtype(df[metric])
    ):
        raise ValueError(
            f"Metric column '{metric}' must be numeric "
            f"for {aggregation} aggregation."
        )

    # Keep rows with a valid group and metric.
    if aggregation == "count":
        valid_df = df.loc[
            df[group_by].notna() & df[metric].notna(),
            [group_by, metric],
        ]
    else:
        numeric_metric = pd.to_numeric(
            df[metric],
            errors="coerce",
        )

        valid_metric = numeric_metric.map(
            lambda value: (
                pd.notna(value)
                and abs(value) != float("inf")
            )
        )

        valid_df = df.loc[
            df[group_by].notna() & valid_metric,
            [group_by, metric],
        ]

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

    if not all(math.isfinite(float(value)) for value in comparison):
        raise ValueError(
            "Aggregation produced a non-finite result. "
            "Check the metric values or choose a different aggregation."
        )

    results = [
        {
            "group": str(group),
            "value": float(value),
        }
        for group, value in comparison.items()
    ]

    # Sort groups from highest to lowest aggregated value.
    results.sort(key=lambda item: item["value"], reverse=True)

    highest_value = results[0]["value"]

    # Include every group tied at the maximum value.
    highest_groups = [
        item["group"]
        for item in results
        if item["value"] == highest_value
    ]

    return {
        "dataset_name": path.name,
        "group_by": group_by,
        "metric": metric,
        "aggregation": aggregation,
        "excluded_rows": int(len(df) - len(valid_df)),
        "group_count": len(results),
        "results": results,
        # Retained for compatibility with existing consumers.
        "highest_group": results[0]["group"],
        "highest_value": highest_value,
        # New field: all groups tied for the maximum.
        "highest_groups": highest_groups,
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
