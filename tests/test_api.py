
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api_service import EvidenceInsufficientError
from main import app



@pytest.fixture
def client():
    """Test HTTP endpoints without starting real dependencies."""
    app.state.embedding_model = Mock(name="embedding_model")
    app.state.qdrant = Mock(name="qdrant")
    app.state.gemini = Mock(name="gemini")
    app.state.startup_error = None

    test_client = TestClient(app)
    yield test_client
    test_client.close()


def test_health_returns_200(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ready_returns_200_when_dependencies_exist(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_returns_503_when_startup_failed(client):
    app.state.startup_error = "Simulated startup failure"

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The application dependencies are not ready."
    )


def test_ask_returns_answer_and_citations(client, monkeypatch):
    mock_answer = Mock(
        return_value={
            "answer": (
                "The total number of employees was 98,844 "
                "in 2025 [Page 8]."
            ),
            "citations": [{"page": 8}],
            "validation": {"passed": True},
            "gate_decision": "ALLOW",
        }
    )

    monkeypatch.setattr(
        "main.answer_question",
        mock_answer,
    )

    response = client.post(
        "/api/v1/ask",
        json={
            "question": "How many employees were there in 2025?"
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "answered"
    assert "[Page 8]" in body["answer"]
    assert body["citations"] == [{"page": 8}]
    assert body["validation_passed"] is True

    mock_answer.assert_called_once()


def test_ask_returns_422_when_evidence_is_insufficient(
    client,
    monkeypatch,
):
    mock_answer = Mock(
        side_effect=EvidenceInsufficientError(
            "Evidence is insufficient."
        )
    )

    monkeypatch.setattr(
        "main.answer_question",
        mock_answer,
    )

    response = client.post(
        "/api/v1/ask",
        json={"question": "What happened?"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["status"] == (
        "insufficient_evidence"
    )
    assert response.json()["detail"]["message"] == (
        "Evidence is insufficient."
    )

    mock_answer.assert_called_once()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": "Hi"},
        {"question": ""},
        {"question": "x" * 1001},
        {"question": 123},
    ],
)
def test_ask_rejects_invalid_request_bodies(client, payload):
    response = client.post(
        "/api/v1/ask",
        json=payload,
    )

    assert response.status_code == 422


def test_ask_returns_503_when_application_is_not_ready(
    client,
    monkeypatch,
):
    app.state.startup_error = "Simulated startup failure"

    mock_answer = Mock()

    monkeypatch.setattr(
        "main.answer_question",
        mock_answer,
    )

    response = client.post(
        "/api/v1/ask",
        json={"question": "How many employees were there?"},
    )

    assert response.status_code == 503
    mock_answer.assert_not_called()
