from pathlib import Path
from typing import Any

import pandas as pd
from langchain.tools import tool


def get_column_statistics(
    csv_path: str,
    column_name: str,
) -> dict[str, Any]:
    """Calculate descriptive statistics for a selected CSV column."""

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

    actual_column_name = column_lookup.get(column_name.casefold())

    if actual_column_name is None:
        raise ValueError(f"Column not found: {column_name}")

    column_name = actual_column_name
    series = df[column_name]
    missing_count = int(series.isna().sum())

    result: dict[str, Any] = {
        "dataset_name": path.name,
        "column_name": column_name,
        "data_type": str(series.dtype),
        "total_rows": len(df),
        "missing_count": missing_count,
        "missing_percentage": round(
            missing_count / len(df) * 100, 2
        ) if len(df) else 0.0,
        "unique_count": int(series.nunique()),
    }

    if (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    ):
        numeric = series.dropna()
        finite_mask = numeric.map(lambda value: pd.notna(value) and abs(value) != float("inf"))
        non_finite_count = int((~finite_mask).sum())
        numeric = numeric[finite_mask]

        result["non_finite_count"] = non_finite_count

        result["statistics"] = None

        if not numeric.empty:
            result["statistics"] = {
                "mean": float(numeric.mean()),
                "median": float(numeric.median()),
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "std": (
                    float(numeric.std())
                    if len(numeric) > 1
                    else None
                ),
                "variance": (
                    float(numeric.var())
                    if len(numeric) > 1
                    else None
                ),
            }

    else:
        values = series.dropna().astype(str)
        result["top_values"] = [
            {"value": str(value), "count": int(count)}
            for value, count in values.value_counts().head(10).items()
        ]

    return result


@tool
def get_column_statistics_tool(
    csv_path: str,
    column_name: str,
) -> dict[str, Any]:
    """Get descriptive statistics for one column in a CSV dataset.

    Use this tool when you need detailed statistics for a specific
    column, including missing values, unique values, numerical
    summaries, or common categorical values.

    Args:
        csv_path: Path to the CSV dataset.
        column_name: Exact name of the column to analyze.
    """
    return get_column_statistics(csv_path, column_name)
