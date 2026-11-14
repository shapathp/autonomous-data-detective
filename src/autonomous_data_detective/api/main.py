import csv
import io
import re
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from autonomous_data_detective.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
)
from autonomous_data_detective.core.config import get_settings
from autonomous_data_detective.graph.graph import (
    build_investigation_graph,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

DATASET_ID_PATTERN = re.compile(r"[0-9a-f]{32}\.csv")

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "An AI-powered data analysis API that uses "
        "a LangGraph workflow to investigate CSV datasets."
    ),
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Check whether the API is running."""
    return {
        "status": "ok",
        "app": settings.app_name,
    }


@app.post("/upload", tags=["Datasets"])
async def upload_csv(
    file: UploadFile = File(...), # noqa: B008 #This tell ruff that File(...) is intensional
) -> dict[str, str]:
    """Validate and store an uploaded CSV dataset."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    if Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=415,
            detail="Only CSV files are supported.",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds the 10 MB limit.",
        )

    try:
        text = content.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text))

        headers = next(reader, None)

        if not headers or not any(header.strip() for header in headers):
            raise ValueError("The CSV must contain a header row.")

        if any(not header.strip() for header in headers):
            raise ValueError("Column names cannot be empty.")

        normalized_headers = [
            header.strip().casefold()
            for header in headers
        ]

        if len(set(normalized_headers)) != len(normalized_headers):
            raise ValueError("Duplicate column names are not supported.")


        data_row_count = 0

        for row in reader:
            # Ignore completely blank lines.
            if not row or not any(field.strip() for field in row):
                continue

            if len(row) != len(headers):
                raise ValueError(
                    "Each data row must have the same number "
                    "of fields as the header."
                )

            data_row_count += 1

        if data_row_count == 0:
            raise ValueError(
                "The CSV must contain at least one data row."
            )


    except (UnicodeDecodeError, csv.Error, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV file: {exc}",
        ) from exc

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    stored_name = f"{uuid4().hex}.csv"
    stored_path = UPLOAD_DIR / stored_name

    stored_path.write_bytes(content)

    return {
        "message": "CSV uploaded successfully.",
        "dataset_id": stored_name,
        "filename": Path(file.filename).name,
        "size_bytes": str(len(content)),
        "columns": ", ".join(headers),
    }


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
)
def analyze_dataset(
    request: AnalyzeRequest,
) -> AnalyzeResponse:
    """Analyze an uploaded CSV using the existing LangGraph workflow."""

    dataset_id = request.dataset_id.strip()
    question = request.question.strip()

    if not dataset_id or not question:
        raise HTTPException(
            status_code=422,
            detail="Dataset ID and question cannot be blank.",
        )

    if not DATASET_ID_PATTERN.fullmatch(dataset_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset ID.",
        )

    upload_root = UPLOAD_DIR.resolve()
    dataset_path = (upload_root / dataset_id).resolve()

    if dataset_path.parent != upload_root:
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset path.",
        )

    if not dataset_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Uploaded dataset not found. Upload the CSV first.",
        )

    try:
        graph = build_investigation_graph()

        result = graph.invoke(
            {
                "user_question": question,
                "csv_path": str(dataset_path),
                "steps_taken": 0,
            }
        )

    except Exception as exc:
        # Keep internal exception details out of the API response.
        raise HTTPException(
            status_code=500,
            detail="The analysis workflow could not be completed.",
        ) from exc

    return AnalyzeResponse(
        dataset_id=dataset_id,
        question=question,
        route=result.get("route", "clarify"),
        final_response=result.get("final_response", ""),
        profile_result=result.get("profile_result"),
        investigation_result=result.get("investigation_result"),
        errors=result.get("errors", []),
        steps_taken=result.get("steps_taken", 0),
    )
