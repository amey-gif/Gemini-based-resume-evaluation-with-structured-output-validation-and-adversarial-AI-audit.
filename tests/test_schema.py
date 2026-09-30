import json

import pytest
from pydantic import ValidationError

from schemas import ResumeEvaluation


@pytest.fixture
def valid_payload():
    return {
        "match_score": 75,
        "top_strengths": [
            "Python experience demonstrated through a project",
            "SQL experience stated in the resume",
        ],
        "missing_skills": ["Docker"],
        "summary": (
            "The resume provides evidence of Python and SQL experience.\n"
            "Docker experience is not evidenced in the supplied resume."
        ),
    }


def test_accepts_valid_json(valid_payload):
    raw_json = json.dumps(valid_payload)

    result = ResumeEvaluation.model_validate_json(raw_json)

    assert result.match_score == 75
    assert len(result.summary.split("\n")) == 2


@pytest.mark.parametrize("score", [-1, 101, "75", 75.5, True, None])
def test_rejects_invalid_scores(valid_payload, score):
    valid_payload["match_score"] = score

    with pytest.raises(ValidationError):
        ResumeEvaluation.model_validate_json(json.dumps(valid_payload))


@pytest.mark.parametrize(
    "summary",
    [
        "Only one line.",
        "First line.\nSecond line.\nThird line.",
        "First line.\n",
        "   \nSecond line.",
        "First line.\nSecond line.\n",
    ],
)
def test_rejects_invalid_summary(valid_payload, summary):
    valid_payload["summary"] = summary

    with pytest.raises(ValidationError):
        ResumeEvaluation.model_validate_json(json.dumps(valid_payload))


def test_rejects_extra_fields(valid_payload):
    valid_payload["candidate_name"] = "Test Candidate"

    with pytest.raises(ValidationError):
        ResumeEvaluation.model_validate_json(json.dumps(valid_payload))


@pytest.mark.parametrize(
    "field",
    ["match_score", "top_strengths", "missing_skills", "summary"],
)
def test_rejects_missing_fields(valid_payload, field):
    del valid_payload[field]

    with pytest.raises(ValidationError):
        ResumeEvaluation.model_validate_json(json.dumps(valid_payload))


@pytest.mark.parametrize("field", ["top_strengths", "missing_skills"])
@pytest.mark.parametrize("items", [[""], ["   "], [123], "Python"])
def test_rejects_invalid_skill_lists(valid_payload, field, items):
    valid_payload[field] = items

    with pytest.raises(ValidationError):
        ResumeEvaluation.model_validate_json(json.dumps(valid_payload))


def test_accepts_empty_lists(valid_payload):
    valid_payload["top_strengths"] = []
    valid_payload["missing_skills"] = []

    result = ResumeEvaluation.model_validate_json(json.dumps(valid_payload))

    assert result.top_strengths == []
    assert result.missing_skills == []


def test_normalizes_windows_summary_newlines(valid_payload):
    valid_payload["summary"] = "First line.\r\nSecond line."

    result = ResumeEvaluation.model_validate_json(json.dumps(valid_payload))

    assert result.summary == "First line.\nSecond line."


def test_rejects_malformed_json():
    with pytest.raises(ValidationError):
        ResumeEvaluation.model_validate_json('{"match_score": 75,')