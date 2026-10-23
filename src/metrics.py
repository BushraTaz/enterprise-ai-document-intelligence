
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST


HTTP_REQUESTS_TOTAL = Counter(
    "enterprise_ai_http_requests_total",
    "Total HTTP requests handled by the API.",
    ["method", "path", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "enterprise_ai_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
)

RATE_LIMIT_REJECTIONS_TOTAL = Counter(
    "enterprise_ai_rate_limit_rejections_total",
    "Total requests rejected by rate limiting.",
)


def render_metrics():
    """Return the current Prometheus metrics exposition."""
    return generate_latest(), CONTENT_TYPE_LATEST
