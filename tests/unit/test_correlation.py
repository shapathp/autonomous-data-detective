import pytest

from autonomous_data_detective.tools.correlation import (
    analyze_correlation,
    analyze_correlation_tool,
)

CSV_PATH = "data/samples/sales.csv"


def test_correlation_returns_expected_fields():
    result = analyze_correlation(
        CSV_PATH, "revenue", "quantity"
    )

    assert result["method"] == "pearson"
    assert result["correlation"] == pytest.approx(-0.4628, abs=0.0001)
    assert result["direction"] == "negative"
    assert result["strength"] == "moderate"
    assert result["valid_pairs"] == 8
    assert result["excluded_rows"] == 2

def test_column_names_are_case_insensitive():
    result = analyze_correlation(
        CSV_PATH, "Revenue", "Quantity"
    )

    assert result["column_a"] == "revenue"
    assert result["column_b"] == "quantity"
    assert result["correlation"] == pytest.approx(
        -0.4628, abs=0.0001
    )


def test_spearman_correlation():
    result = analyze_correlation(
        CSV_PATH, "revenue", "quantity", "spearman"
    )

    assert -1.0 <= result["correlation"] <= 1.0
    assert result["method"] == "spearman"


def test_missing_column_is_rejected():
    with pytest.raises(ValueError, match="Column not found"):
        analyze_correlation(
            CSV_PATH, "revenue", "profit"
        )


def test_same_column_is_rejected():
    with pytest.raises(ValueError, match="different columns"):
        analyze_correlation(
            CSV_PATH, "revenue", "revenue"
        )


def test_nonnumeric_column_is_rejected():
    with pytest.raises(ValueError, match="must be numeric"):
        analyze_correlation(
            CSV_PATH, "region", "revenue"
        )


def test_unsupported_method_is_rejected():
    with pytest.raises(ValueError, match="Unsupported correlation"):
        analyze_correlation(
            CSV_PATH, "revenue", "quantity", "kendall"
        )


def test_constant_column_is_rejected(tmp_path):
    csv_file = tmp_path / "constant.csv"
    csv_file.write_text(
        "x,y\n1,2\n1,3\n1,4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="constant column"):
        analyze_correlation(str(csv_file), "x", "y")


def test_langchain_tool_can_be_invoked():
    result = analyze_correlation_tool.invoke(
        {
            "csv_path": CSV_PATH,
            "column_a": "revenue",
            "column_b": "quantity",
        }
    )

    assert result["direction"] == "negative"
    assert result["valid_pairs"] == 8

def test_non_finite_values_are_excluded(tmp_path):
    csv_file = tmp_path / "non_finite.csv"
    csv_file.write_text(
        "x,y\n"
        "1,10\n"
        "2,20\n"
        "inf,30\n"
        "4,-inf\n"
        "5,50\n",
        encoding="utf-8",
    )

    result = analyze_correlation(
        str(csv_file),
        "x",
        "y",
    )

    assert result["correlation"] == pytest.approx(1.0)
    assert result["valid_pairs"] == 3
    assert result["excluded_rows"] == 2


def test_non_csv_file_is_rejected(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("x,y\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Only CSV files are supported"):
        analyze_correlation(str(file_path), "x", "y")


def test_missing_csv_file_is_rejected(tmp_path):
    file_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        analyze_correlation(str(file_path), "x", "y")


def test_empty_dataset_is_rejected(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("x,y\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset is empty"):
        analyze_correlation(str(file_path), "x", "y")


def test_missing_first_column_is_rejected():
    with pytest.raises(ValueError, match="Column not found: profit"):
        analyze_correlation(CSV_PATH, "profit", "quantity")


def test_insufficient_complete_pairs_are_rejected(tmp_path):
    file_path = tmp_path / "insufficient.csv"
    file_path.write_text(
        "x,y\n1,10\n2,\n,30\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="At least two complete row pairs are required",
    ):
        analyze_correlation(str(file_path), "x", "y")




def test_nan_correlation_result_is_rejected(monkeypatch):
    import pandas as pd

    def return_nan(self, other, method="pearson", min_periods=None):
        return float("nan")

    monkeypatch.setattr(pd.Series, "corr", return_nan)

    with pytest.raises(
        ValueError,
        match="Correlation could not be calculated",
    ):
        analyze_correlation(
            CSV_PATH,
            "revenue",
            "quantity",
        )

def test_infinite_correlation_result_is_rejected(monkeypatch):
    import pandas as pd

    def return_infinity(
        self, other, method="pearson", min_periods=None
    ):
        return float("inf")

    monkeypatch.setattr(pd.Series, "corr", return_infinity)

    with pytest.raises(
        ValueError,
        match="Correlation could not be calculated",
    ):
        analyze_correlation(
            CSV_PATH,
            "revenue",
            "quantity",
        )

def test_non_string_method_is_rejected():
    with pytest.raises(ValueError, match="Unsupported correlation method"):
        analyze_correlation(
            CSV_PATH,
            "revenue",
            "quantity",
            method=123,
        )