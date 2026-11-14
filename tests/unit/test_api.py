from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from autonomous_data_detective.api import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a test client with isolated upload storage."""
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)

    with TestClient(main.app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": main.settings.app_name,
    }


def test_upload_valid_csv(client):
    content = (
        b"order_id,region,product,revenue\n"
        b"1,North,Laptop,1000\n"
        b"2,South,Phone,500\n"
    )

    response = client.post(
        "/upload",
        files={
            "file": (
                "sales.csv",
                content,
                "text/csv",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["message"] == "CSV uploaded successfully."
    assert data["filename"] == "sales.csv"
    assert data["columns"] == "order_id, region, product, revenue"
    assert data["size_bytes"] == str(len(content))
    assert len(data["dataset_id"]) == 36
    assert data["dataset_id"].endswith(".csv")
    assert (main.UPLOAD_DIR / data["dataset_id"]).read_bytes() == content


def test_upload_rejects_non_csv_file(client):
    response = client.post(
        "/upload",
        files={
            "file": (
                "sales.txt",
                b"some text",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
    assert response.json()["detail"] == "Only CSV files are supported."


def test_upload_rejects_empty_file(client):
    response = client.post(
        "/upload",
        files={
            "file": (
                "empty.csv",
                b"",
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty."


def test_upload_rejects_invalid_csv(client):
    response = client.post(
        "/upload",
        files={
            "file": (
                "invalid.csv",
                b",,,\n1,2,3,4\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
    assert "Invalid CSV file" in response.json()["detail"]

def test_upload_rejects_empty_column_name(client):
    response = client.post(
        "/upload",
        files={
            "file": (
                "empty_column.csv",
                b"product,,revenue\nLaptop,Electronics,1000\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
    assert "Column names cannot be empty" in response.json()["detail"]


def test_upload_rejects_duplicate_column_names(client):
    response = client.post(
        "/upload",
        files={
            "file": (
                "duplicate_columns.csv",
                b"product,revenue,revenue\nLaptop,1000,1200\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
    assert "Duplicate column names" in response.json()["detail"]

def test_upload_rejects_inconsistent_row_lengths(client):
    content = (
        b"product,revenue\n"
        b"Laptop,1000,unexpected\n"
        b"Phone,500\n"
    )

    response = client.post(
        "/upload",
        files={
            "file": (
                "inconsistent.csv",
                content,
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
    assert "Invalid CSV file" in response.json()["detail"]



def test_upload_rejects_csv_without_data_rows(client):
    response = client.post(
        "/upload",
        files={
            "file": (
                "headers_only.csv",
                b"order_id,region,revenue\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
    assert "at least one data row" in response.json()["detail"]


def test_upload_rejects_file_over_size_limit(client):
    content = b"a" * (main.MAX_UPLOAD_BYTES + 1)

    response = client.post(
        "/upload",
        files={
            "file": (
                "large.csv",
                content,
                "text/csv",
            )
        },
    )

    assert response.status_code == 413
    assert "10 MB limit" in response.json()["detail"]


def test_analyze_success(client, monkeypatch):
    dataset_id = "a" * 32 + ".csv"
    dataset_path = main.UPLOAD_DIR / dataset_id
    dataset_path.write_text(
        "product,revenue\nLaptop,1000\nPhone,500\n",
        encoding="utf-8",
    )

    expected_result = {
        "route": "investigate",
        "final_response": "Laptop has the highest revenue.",
        "profile_result": None,
        "investigation_result": "Laptop: 1000",
        "errors": [],
        "steps_taken": 1,
    }

    fake_graph = Mock()
    fake_graph.invoke.return_value = expected_result

    fake_builder = Mock(return_value=fake_graph)
    monkeypatch.setattr(
        main,
        "build_investigation_graph",
        fake_builder,
    )

    response = client.post(
        "/analyze",
        json={
            "dataset_id": dataset_id,
            "question": "Which product has the highest revenue?",
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["dataset_id"] == dataset_id
    assert data["route"] == "investigate"
    assert data["final_response"] == (
        "Laptop has the highest revenue."
    )
    assert data["investigation_result"] == "Laptop: 1000"
    assert data["errors"] == []
    assert data["steps_taken"] == 1

    fake_builder.assert_called_once_with()
    fake_graph.invoke.assert_called_once_with(
        {
            "user_question": "Which product has the highest revenue?",
            "csv_path": str(dataset_path.resolve()),
            "steps_taken": 0,
        }
    )


def test_analyze_rejects_invalid_dataset_id(client):
    response = client.post(
        "/analyze",
        json={
            "dataset_id": "sales.csv",
            "question": "Which product has the highest revenue?",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid dataset ID."


def test_analyze_returns_404_for_missing_dataset(client):
    response = client.post(
        "/analyze",
        json={
            "dataset_id": "b" * 32 + ".csv",
            "question": "Which product has the highest revenue?",
        },
    )

    assert response.status_code == 404
    assert "Upload the CSV first" in response.json()["detail"]


def test_analyze_rejects_blank_question(client):
    response = client.post(
        "/analyze",
        json={
            "dataset_id": "a" * 32 + ".csv",
            "question": "   ",
        },
    )

    assert response.status_code == 422


def test_analyze_returns_500_when_graph_fails(
    client,
    monkeypatch,
):
    dataset_id = "c" * 32 + ".csv"
    (main.UPLOAD_DIR / dataset_id).write_text(
        "product,revenue\nLaptop,1000\n",
        encoding="utf-8",
    )

    fake_graph = Mock()
    fake_graph.invoke.side_effect = RuntimeError(
        "Internal test failure"
    )

    monkeypatch.setattr(
        main,
        "build_investigation_graph",
        Mock(return_value=fake_graph),
    )

    response = client.post(
        "/analyze",
        json={
            "dataset_id": dataset_id,
            "question": "Which product has the highest revenue?",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "The analysis workflow could not be completed."
    )
    assert "Internal test failure" not in response.text
