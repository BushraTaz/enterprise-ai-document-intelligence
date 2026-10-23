
from fastapi.testclient import TestClient

import main
from rate_limiter import InMemoryRateLimiter


def test_rate_limit_returns_429_without_calling_service(monkeypatch):
    # Allow only one request per client in this test.
    monkeypatch.setattr(
        main,
        "ask_rate_limiter",
        InMemoryRateLimiter(max_requests=1, window_seconds=60),
    )

    # Keep the test independent of Qdrant and Gemini.
    service_calls = []

    def fake_answer_question(**kwargs):
        service_calls.append(kwargs)
        return {
            "answer": "Volvo Group reported 98,844 employees. [Page 8]",
            "citations": [{"page": 8}],
            "validation": {"passed": True},
            "gate_decision": "ALLOW",
        }

    monkeypatch.setattr(
        main,
        "answer_question",
        fake_answer_question,
    )

    # Set mock application state without starting external services.
    main.app.state.startup_error = None
    main.app.state.embedding_model = object()
    main.app.state.qdrant = object()
    main.app.state.gemini = object()

    client = TestClient(main.app)

    payload = {
        "question": "How many employees did Volvo Group have in 2025?"
    }

    first_response = client.post("/api/v1/ask", json=payload)

    assert first_response.status_code == 200
    assert first_response.headers.get("X-Request-ID")
    assert len(service_calls) == 1

    second_response = client.post("/api/v1/ask", json=payload)

    assert second_response.status_code == 429
    assert second_response.headers.get("Retry-After")
    assert second_response.headers.get("X-Request-ID")
    assert second_response.json()["detail"]["status"] == (
        "rate_limit_exceeded"
    )

    # The rejected request must not call the answer service.
    assert len(service_calls) == 1
