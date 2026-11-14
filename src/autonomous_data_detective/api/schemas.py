from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    dataset_id: str = Field(
        min_length=1,
        description="Dataset ID returned by POST /upload.",
    )
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural-language question about the CSV dataset.",
    )


class AnalyzeResponse(BaseModel):
    dataset_id: str
    question: str
    route: str
    final_response: str
    profile_result: str | None = None
    investigation_result: str | None = None
    errors: list[str] = Field(default_factory=list)
    steps_taken: int = 0
