from pathlib import Path
from typing import Any

import pandas as pd
from langchain.tools import tool



def profile_dataset(csv_path: str) -> dict[str, Any]:
    """Profile a CSV file and return structured dataset evidence."""

    path = Path(csv_path)

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path.name}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The CSV file contains no data rows.")

    if len(df.columns) == 0:
        raise ValueError("The CSV file contains no columns.")

    if len(df.columns) > 200:
        raise ValueError("The dataset exceeds the 200-column limit.")

    columns: dict[str, Any] = {}

    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())

        column_info: dict[str, Any] = {
            "data_type": str(series.dtype),
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_count / len(df) * 100, 2
            ),
            "unique_count": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            numeric = series.dropna()

            if not numeric.empty:
                column_info["statistics"] = {
                    "mean": float(numeric.mean()),
                    "median": float(numeric.median()),
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "std": (
                        float(numeric.std())
                        if len(numeric) > 1
                        else None
                    ),
                }
            else:
                column_info["statistics"] = None

        else:
            values = series.dropna().astype(str)
            column_info["top_values"] = [
                {"value": str(value), "count": int(count)}
                for value, count in values.value_counts().head(5).items()
            ]

        columns[str(column)] = column_info

    duplicate_count = int(df.duplicated().sum())

    return {
        "dataset_name": path.name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_row_count": duplicate_count,
        "columns": columns,
    }




# LangChain tool wrapper
@tool
def profile_dataset_tool(csv_path: str) -> dict:
    """Profile a CSV dataset.

    Returns row and column counts, duplicate rows, missing-value
    percentages, column types, unique counts, and descriptive
    statistics. Use this tool when you need to understand a dataset
    before investigating a business or data-quality question.

    Args:
        csv_path: Path to the CSV file within the project's approved
            sample-data directory.
    """
    return profile_dataset(csv_path)
