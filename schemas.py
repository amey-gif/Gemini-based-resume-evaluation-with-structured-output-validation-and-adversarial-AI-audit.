from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)


# Each list item must contain text, not an empty or whitespace-only string.
NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ResumeEvaluation(BaseModel):
    """Validated output from the resume evaluation pipeline."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    match_score: int = Field(
        ge=0,
        le=100,
        description="Integer match score between 0 and 100.",
    )

    top_strengths: list[NonEmptyText] = Field(
        description=(
            "Job-relevant strengths supported by the supplied resume. "
            "Use an empty list if no strengths are evidenced."
        ),
    )

    missing_skills: list[NonEmptyText] = Field(
        description=(
            "Skills explicitly required by the job description but not "
            "evidenced in the resume. Use an empty list if none are missing."
        ),
    )

    summary: str = Field(
        description=(
            "Exactly two nonempty lines separated by one newline character. "
            "Line 1 summarizes evidenced fit. "
            "Line 2 summarizes gaps or limitations."
        ),
    )

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        # Normalize Windows line endings before checking the two-line rule.
        normalized = value.replace("\r\n", "\n")

        if "\r" in normalized:
            raise ValueError("Summary contains an unsupported line separator.")

        lines = normalized.split("\n")

        if len(lines) != 2:
            raise ValueError("Summary must contain exactly two lines.")

        if any(not line.strip() for line in lines):
            raise ValueError("Both summary lines must contain text.")

        return "\n".join(line.strip() for line in lines)