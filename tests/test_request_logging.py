
import json
import logging
from unittest.mock import Mock
from uuid import UUID

from fastapi.testclient import TestClient

import main
from main import app


import pytest
import main


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Prevent rate-limit state from leaking between tests."""
    main.ask_rate_limiter.reset()
    yield
    main.ask_rate_limiter.reset()

def create_test_client():
    """Create a client without running application startup."""
    app.state.embedding_model = Mock(name="embedding_model")
    app.state.qdrant = Mock(name="qdrant")
    app.state.gemini = Mock(name="gemini")
    app.state.startup_error = None

    return TestClient(app)


def test_request_id_is_returned_and_logged(monkeypatch, caplog):
    """The response ID should match the structured request log."""
    with caplog.at_level(logging.INFO, logger="enterprise_ai_api"):
        client = create_test_client()
        try:
            response = client.get("/health")
        finally:
            client.close()

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

    request_id = response.headers["X-Request-ID"]
    UUID(request_id)

    matching_records = [
        record
        for record in caplog.records
        if record.name == "enterprise_ai_api"
        and record.getMessage() == "http_request_completed"
    ]

    assert len(matching_records) == 1

    record = matching_records[0]
    assert record.request_id == request_id
    assert record.method == "GET"
    assert record.path == "/health"
    assert record.status_code == 200
    assert record.duration_ms >= 0


def test_each_request_gets_a_new_request_id():
    """Separate requests should receive distinct UUIDs."""
    client = create_test_client()

    try:
        first_response = client.get("/health")
        second_response = client.get("/health")
    finally:
        client.close()

    first_id = first_response.headers["X-Request-ID"]
    second_id = second_response.headers["X-Request-ID"]

    UUID(first_id)
    UUID(second_id)

    assert first_id != second_id


def test_question_failure_is_logged_without_exposing_exception(
    monkeypatch,
    caplog,
):
    """Unexpected service errors should be logged and safely returned."""
    service_mock = Mock(
        side_effect=RuntimeError("sensitive internal detail")
    )
    monkeypatch.setattr(main, "answer_question", service_mock)

    client = create_test_client()

    try:
        with caplog.at_level(
            logging.ERROR,
            logger="enterprise_ai_api",
        ):
            response = client.post(
                "/api/v1/ask",
                json={
                    "question": "How many employees were there?"
                },
            )
    finally:
        client.close()

    assert response.status_code == 500
    assert "X-Request-ID" in response.headers
    assert "sensitive internal detail" not in response.text

    request_id = response.headers["X-Request-ID"]
    UUID(request_id)

    error_records = [
        record
        for record in caplog.records
        if record.name == "enterprise_ai_api"
        and record.getMessage() == "question_answering_failed"
    ]

    assert len(error_records) == 1
    assert error_records[0].request_id == request_id
    assert error_records[0].method == "POST"
    assert error_records[0].path == "/api/v1/ask"

    service_mock.assert_called_once()
