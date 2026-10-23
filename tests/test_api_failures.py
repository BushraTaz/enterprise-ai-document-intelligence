
from fastapi.testclient import TestClient

import main


def configure_ready_app():
    """Set mock dependencies without initializing external services."""
    main.app.state.startup_error = None
    main.app.state.embedding_model = object()
    main.app.state.qdrant = object()
    main.app.state.gemini = object()


def test_readiness_returns_503_when_startup_failed(monkeypatch):
    monkeypatch.setattr(
        main,
        "answer_question",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Answer service must not be called")
        ),
    )

    main.app.state.startup_error = "Dependency initialization failed"
    client = TestClient(main.app)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The application dependencies are not ready."
    )


def test_ask_returns_503_when_startup_failed(monkeypatch):
    monkeypatch.setattr(
        main,
        "answer_question",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Answer service must not be called")
        ),
    )

    main.app.state.startup_error = "Dependency initialization failed"
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/ask",
        json={"question": "How many employees did Volvo Group have?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The application dependencies are not ready."
    )


def test_unexpected_service_error_returns_generic_500(monkeypatch):
    configure_ready_app()

    def failing_service(**kwargs):
        raise RuntimeError("Internal implementation details")

    monkeypatch.setattr(main, "answer_question", failing_service)
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/ask",
        json={"question": "How many employees did Volvo Group have?"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "An unexpected error occurred while processing the question."
    )
    assert "Internal implementation details" not in response.text
    assert response.headers.get("X-Request-ID")


def test_health_endpoint_remains_available_when_startup_failed():
    main.app.state.startup_error = "Dependency initialization failed"
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
