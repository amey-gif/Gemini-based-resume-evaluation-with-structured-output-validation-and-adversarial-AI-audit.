from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.genai import types
from pydantic import ValidationError

from evaluate_resume import evaluate_resume, sanitize_json_response


@pytest.fixture
def valid_json():
    return (
        '{"match_score": 75,'
        '"top_strengths": ["Python project experience"],'
        '"missing_skills": ["Docker"],'
        '"summary": "Python experience is evidenced.\\n'
        'Docker is not evidenced."}'
    )


def mock_client(text, finish_reason=types.FinishReason.STOP):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
        text=text,
    )
    return client


def test_accepts_valid_model_response(valid_json):
    client = mock_client(valid_json)

    result = evaluate_resume(
        client,
        "test-model",
        "Skills: Python",
        "Required skills: Python and Docker",
    )

    assert result.match_score == 75
    assert result.missing_skills == ["Docker"]
    client.models.generate_content.assert_called_once()


def test_accepts_whole_response_json_fence(valid_json):
    wrapped = f"```json\n{valid_json}\n```"

    assert sanitize_json_response(wrapped) == valid_json


def test_rejects_preamble_before_json(valid_json):
    client = mock_client("Here is your evaluation:\n" + valid_json)

    with pytest.raises(ValidationError):
        evaluate_resume(client, "test-model", "Python", "Python")


@pytest.mark.parametrize(
    "resume,jd",
    [
        ("", "Python"),
        ("   ", "Python"),
        ("Python", ""),
        ("Python", "   "),
    ],
)
def test_empty_inputs_do_not_call_api(resume, jd):
    client = Mock()

    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_resume(client, "test-model", resume, jd)

    client.models.generate_content.assert_not_called()


def test_rejects_empty_response():
    client = mock_client("")

    with pytest.raises(ValueError, match="empty response"):
        evaluate_resume(client, "test-model", "Python", "Python")


def test_rejects_missing_candidates():
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        candidates=[],
        text=None,
    )

    with pytest.raises(ValueError, match="no candidate"):
        evaluate_resume(client, "test-model", "Python", "Python")


def test_rejects_truncated_generation(valid_json):
    client = mock_client(
        valid_json,
        finish_reason=types.FinishReason.MAX_TOKENS,
    )

    with pytest.raises(ValueError, match="did not complete normally"):
        evaluate_resume(client, "test-model", "Python", "Python")


def test_rejects_malformed_json():
    client = mock_client('{"match_score": 75,')

    with pytest.raises(ValidationError):
        evaluate_resume(client, "test-model", "Python", "Python")


def test_rejects_schema_invalid_response(valid_json):
    invalid_json = valid_json.replace(
        '"match_score": 75',
        '"match_score": 175',
    )
    client = mock_client(invalid_json)

    with pytest.raises(ValidationError):
        evaluate_resume(client, "test-model", "Python", "Python")