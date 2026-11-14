from pathlib import Path
from typing import Any

import pandas as pd
from langchain.tools import tool


def analyze_missing_values(csv_path: str) -> dict[str, Any]:
    """Analyze missing values across a CSV dataset."""

    path = Path(csv_path)

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path.name}")

    df = pd.read_csv(path)

    if df.empty or len(df.columns) == 0:
        raise ValueError("The dataset is empty.")

    total_cells = int(df.shape[0] * df.shape[1])
    total_missing = int(df.isna().sum().sum())

    missing_by_column = []

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_percentage = round(
            missing_count / len(df) * 100, 2
        )

        if missing_count == 0:
            recommendation = "No missing values detected."
        elif missing_percentage >= 50:
            recommendation = (
                "Investigate whether the column should be retained; "
                "consider dropping it only if it is not essential."
            )
        else:
            recommendation = (
                "Investigate the cause of missingness; consider "
                "imputation or reviewing affected records."
            )

        missing_by_column.append(
            {
                "column": str(column),
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
                "recommendation": recommendation,
            }
        )

    columns_with_missing = sum(
        item["missing_count"] > 0
        for item in missing_by_column
    )

    return {
        "dataset_name": path.name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "total_cells": total_cells,
        "total_missing_values": total_missing,
        "missing_cell_percentage": round(
            total_missing / total_cells * 100, 2
        ),
        "columns_with_missing_values": columns_with_missing,
        "columns": missing_by_column,
    }


@tool
def analyze_missing_values_tool(csv_path: str) -> dict[str, Any]:
    """Analyze missing values and suggest handling strategies.

    Use this tool when investigating data quality or determining
    which columns and records contain missing values.

    Args:
        csv_path: Path to the CSV dataset.
    """
    return analyze_missing_values(csv_path)
