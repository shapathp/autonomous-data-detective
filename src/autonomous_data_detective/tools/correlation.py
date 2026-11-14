import math
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.tools import tool


def analyze_correlation(
    csv_path: str,
    column_a: str,
    column_b: str,
    method: str = "pearson",
) -> dict[str, Any]:
    """Measure the correlation between two numerical columns."""

    path = Path(csv_path)

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path.name}")

    df = pd.read_csv(path)

    if df.empty or len(df.columns) == 0:
        raise ValueError("The dataset is empty.")

    column_lookup = {
        column.casefold(): column
        for column in df.columns
    }

    actual_column_a = column_lookup.get(column_a.casefold())
    actual_column_b = column_lookup.get(column_b.casefold())

    if actual_column_a is None:
        raise ValueError(f"Column not found: {column_a}")

    if actual_column_b is None:
        raise ValueError(f"Column not found: {column_b}")

    column_a = actual_column_a
    column_b = actual_column_b

    if column_a == column_b:
        raise ValueError("Choose two different columns.")

    if method not in {"pearson", "spearman"}:
        raise ValueError(
            "Unsupported correlation method. "
            "Choose 'pearson' or 'spearman'."
        )

    for column in (column_a, column_b):
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must be numeric."
            )

    paired = df[[column_a, column_b]].dropna()
    finite_mask = (
        paired[column_a].map(lambda value: abs(value) != float("inf"))
        & paired[column_b].map(lambda value: abs(value) != float("inf"))
    )
    paired = paired.loc[finite_mask]
    valid_pairs = len(paired)
    excluded_rows = len(df) - valid_pairs

    if valid_pairs < 2:
        raise ValueError(
            "At least two complete row pairs are required."
        )

    if (
        paired[column_a].nunique() < 2
        or paired[column_b].nunique() < 2
    ):
        raise ValueError(
            "Correlation cannot be calculated for a constant column."
        )

    correlation = paired[column_a].corr(
        paired[column_b],
        method=method,
    )

    if pd.isna(correlation):
        raise ValueError(
            "Correlation could not be calculated from these values."
        )

    value = float(correlation)

    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(
            "Correlation could not be calculated from these values."
        )

    if abs(value) < 0.3:
        strength = "weak"
    elif abs(value) < 0.7:
        strength = "moderate"
    else:
        strength = "strong"

    direction = (
        "positive" if value > 0
        else "negative" if value < 0
        else "none"
    )

    return {
        "dataset_name": path.name,
        "column_a": column_a,
        "column_b": column_b,
        "method": method,
        "correlation": round(value, 4),
        "strength": strength,
        "direction": direction,
        "valid_pairs": valid_pairs,
        "excluded_rows": excluded_rows,
        "interpretation": (
            f"The columns have a {strength} {direction} "
            f"correlation ({value:.4f}). Correlation alone "
            "does not establish causation."
        ),
    }


@tool
def analyze_correlation_tool(
    csv_path: str,
    column_a: str,
    column_b: str,
    method: str = "pearson",
) -> dict[str, Any]:
    """Analyze the correlation between two numeric CSV columns.

    Use Pearson correlation for linear relationships or Spearman
    correlation for monotonic relationships based on ranks.

    Args:
        csv_path: Path to the CSV dataset.
        column_a: Name of the first numeric column.
        column_b: Name of the second numeric column.
        method: Either 'pearson' or 'spearman'.
    """
    return analyze_correlation(
        csv_path, column_a, column_b, method
    )
