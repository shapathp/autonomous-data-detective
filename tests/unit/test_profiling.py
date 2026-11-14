import pandas as pd
import pytest

from autonomous_data_detective.tools.profiling import profile_dataset


def test_profile_dataset_returns_expected_statistics(tmp_path):
    csv_file = tmp_path / "sales.csv"

    df = pd.DataFrame(
        {
            "sales": [100, 200, 300],
            "region": ["North", "South", "North"],
        }
    )
    df.to_csv(csv_file, index=False)

    result = profile_dataset(str(csv_file))

    assert result["dataset_name"] == "sales.csv"
    assert result["row_count"] == 3
    assert result["column_count"] == 2
    assert result["duplicate_row_count"] == 0

    assert result["columns"]["sales"]["statistics"]["mean"] == 200.0
    assert result["columns"]["sales"]["statistics"]["median"] == 200.0

    assert result["columns"]["region"]["unique_count"] == 2


def test_profile_dataset_detects_missing_values(tmp_path):
    csv_file = tmp_path / "missing.csv"

    pd.DataFrame(
        {
            "sales": [100, None, 300],
        }
    ).to_csv(csv_file, index=False)

    result = profile_dataset(str(csv_file))

    sales = result["columns"]["sales"]

    assert sales["missing_count"] == 1
    assert sales["missing_percentage"] == pytest.approx(33.33)


def test_profile_dataset_rejects_non_csv(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("not a CSV file")

    with pytest.raises(ValueError, match="Only CSV files"):
        profile_dataset(str(file_path))


def test_profile_dataset_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        profile_dataset(str(tmp_path / "missing.csv"))

def test_profile_dataset_handles_duplicate_column_names(tmp_path):
    csv_file = tmp_path / "duplicate_columns.csv"
    csv_file.write_text(
        "sales,sales\n100,200\n300,400\n",
        encoding="utf-8",
    )

    result = profile_dataset(str(csv_file))

    assert result["column_count"] == 2
    assert len(result["columns"]) == 2

def test_profile_dataset_rejects_empty_csv(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("", encoding="utf-8")

    with pytest.raises(pd.errors.EmptyDataError):
        profile_dataset(str(csv_file))


def test_profile_dataset_rejects_header_only_csv(tmp_path):
    csv_file = tmp_path / "header_only.csv"
    csv_file.write_text("sales,region\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="The CSV file contains no data rows",
    ):
        profile_dataset(str(csv_file))

def test_profile_dataset_handles_all_missing_numeric_column(tmp_path):
    csv_file = tmp_path / "all_missing.csv"

    pd.DataFrame(
        {
            "sales": [None, None, None],
        }
    ).to_csv(csv_file, index=False)

    result = profile_dataset(str(csv_file))

    sales = result["columns"]["sales"]

    assert sales["missing_count"] == 3
    assert sales["missing_percentage"] == 100.0
    assert sales["statistics"] is None

def test_profile_dataset_counts_duplicate_rows(tmp_path):
    csv_file = tmp_path / "duplicates.csv"

    pd.DataFrame(
        {
            "sales": [100, 200, 100, 100],
            "region": ["North", "South", "North", "North"],
        }
    ).to_csv(csv_file, index=False)

    result = profile_dataset(str(csv_file))

    assert result["row_count"] == 4
    assert result["column_count"] == 2
    assert result["duplicate_row_count"] == 2

def test_profile_dataset_rejects_more_than_200_columns(tmp_path):
    csv_file = tmp_path / "too_many_columns.csv"

    columns = [f"column_{i}" for i in range(201)]
    df = pd.DataFrame([[1] * 201], columns=columns)
    df.to_csv(csv_file, index=False)

    with pytest.raises(
        ValueError,
        match="The dataset exceeds the 200-column limit",
    ):
        profile_dataset(str(csv_file))




def test_csv_with_no_columns_is_rejected(tmp_path, monkeypatch):
    from unittest.mock import Mock

    import pandas as pd

    from autonomous_data_detective.tools.profiling import profile_dataset

    file_path = tmp_path / "no_columns.csv"
    file_path.write_text("placeholder", encoding="utf-8")

    fake_df = Mock()
    fake_df.empty = False
    fake_df.columns = []

    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda path: fake_df,
    )

    with pytest.raises(
        ValueError,
        match="The CSV file contains no columns",
    ):
        profile_dataset(str(file_path))

def test_profile_dataset_excludes_non_finite_numeric_values(tmp_path):
    csv_file = tmp_path / "non_finite.csv"
    csv_file.write_text(
        "revenue\n10\n20\ninf\n-inf\n",
        encoding="utf-8",
    )

    result = profile_dataset(str(csv_file))
    revenue = result["columns"]["revenue"]

    assert revenue["statistics"]["mean"] == 15.0
    assert revenue["statistics"]["min"] == 10.0
    assert revenue["statistics"]["max"] == 20.0
    assert revenue["non_finite_count"] == 2
