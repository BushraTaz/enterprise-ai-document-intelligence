
from fastapi.testclient import TestClient

import main
from metrics import (
    HTTP_REQUESTS_TOTAL,
    RATE_LIMIT_REJECTIONS_TOTAL,
)


def test_metrics_endpoint_returns_prometheus_format():
    client = TestClient(main.app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "enterprise_ai_http_requests_total" in response.text
    assert (
        "enterprise_ai_http_request_duration_seconds"
        in response.text
    )
    assert (
        "enterprise_ai_rate_limit_rejections_total"
        in response.text
    )


def test_request_metrics_record_health_request():
    client = TestClient(main.app)

    before = HTTP_REQUESTS_TOTAL.labels(
        method="GET",
        path="/health",
        status_code="200",
    )._value.get()

    response = client.get("/health")

    after = HTTP_REQUESTS_TOTAL.labels(
        method="GET",
        path="/health",
        status_code="200",
    )._value.get()

    assert response.status_code == 200
    assert after == before + 1


def test_metrics_endpoint_is_not_rate_limited():
    client = TestClient(main.app)

    # Metrics scraping must not depend on the ask endpoint's limit.
    for _ in range(3):
        response = client.get("/metrics")
        assert response.status_code == 200
