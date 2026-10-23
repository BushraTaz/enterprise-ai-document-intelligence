
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from google import genai

from api_service import EvidenceInsufficientError, answer_question
from rate_limiter import InMemoryRateLimiter
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi.responses import Response

from metrics import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    RATE_LIMIT_REJECTIONS_TOTAL,
    render_metrics,
)


load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"


class JsonFormatter(logging.Formatter):
    """Format each log record as a JSON object."""

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(
                record,
                datefmt="%Y-%m-%dT%H:%M:%S",
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in (
            "request_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
        ):
            value = getattr(record, field, None)
            if value is not None:
                log_entry[field] = value

        if record.exc_info:
            log_entry["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(log_entry, ensure_ascii=False)


def configure_logging():
    """Configure JSON logging without duplicate handlers."""
    root_logger = logging.getLogger()

    if not any(
        getattr(handler, "_enterprise_json_handler", False)
        for handler in root_logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._enterprise_json_handler = True
        root_logger.addHandler(handler)

    root_logger.setLevel(logging.INFO)


configure_logging()
logger = logging.getLogger("enterprise_ai_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared dependencies once at startup."""
    app.state.embedding_model = None
    app.state.qdrant = None
    app.state.gemini = None
    app.state.startup_error = None

    try:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured"
            )

        app.state.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )
        app.state.qdrant = QdrantClient(url=QDRANT_URL)
        app.state.gemini = genai.Client(api_key=api_key)

        logger.info("application_dependencies_initialized")

    except Exception:
        logger.exception("application_startup_failed")
        app.state.startup_error = "Dependency initialization failed"

    yield

    if app.state.qdrant is not None:
        app.state.qdrant.close()
        logger.info("qdrant_client_closed")


app = FastAPI(
    title="Enterprise AI Document Intelligence API",
    description="A document question-answering API with evidence checks.",
    version="0.1.0",
    lifespan=lifespan,
)



# Rate limiter: maximum 10 ask requests per client per 60 seconds.
ask_rate_limiter = InMemoryRateLimiter(
    max_requests=10,
    window_seconds=60,
)


def enforce_ask_rate_limit(request: Request):
    """Reject requests that exceed the configured limit."""
    client_key = (
        request.client.host
        if request.client is not None
        else "unknown"
    )

    retry_after = ask_rate_limiter.check(client_key)

    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "status": "rate_limit_exceeded",
                "message": "Too many requests. Please try again later.",
            },
            headers={"Retry-After": str(retry_after)},
        )


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        description="A question about the indexed company documents.",
    )


class Citation(BaseModel):
    page: int


class AskResponse(BaseModel):
    status: str
    answer: str
    citations: list[Citation]
    validation_passed: bool


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Attach a request ID and log the final HTTP outcome."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    started_at = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_middleware_failed")
        response = JSONResponse(
            status_code=500,
            content={
                "detail": "An unexpected internal error occurred.",
                "request_id": request_id,
            },
        )

    response.headers["X-Request-ID"] = request_id

    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        path=request.url.path,
        status_code=str(response.status_code),
    ).inc()

    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=request.method,
        path=request.url.path,
    ).observe(time.perf_counter() - started_at)

    if (
        request.url.path == "/api/v1/ask"
        and response.status_code == 429
    ):
        RATE_LIMIT_REJECTIONS_TOTAL.inc()

    logger.info(
        "http_request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(
                (time.perf_counter() - started_at) * 1000,
                2,
            ),
        },
    )

    return response

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "static"


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")
@app.get("/health")
def health():
    """Check whether the API process is running."""
    return {"status": "healthy"}


@app.get("/ready")
def readiness(request: Request):
    """Check whether the required dependencies initialized."""
    if request.app.state.startup_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The application dependencies are not ready.",
        )

    if any(
        getattr(request.app.state, name, None) is None
        for name in ("embedding_model", "qdrant", "gemini")
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The application dependencies are not ready.",
        )

    return {"status": "ready"}


@app.post(
    "/api/v1/ask",
    response_model=AskResponse,
    dependencies=[Depends(enforce_ask_rate_limit)],
)
def ask_question(payload: QuestionRequest, request: Request):
    """Answer a question when retrieved evidence passes the gate."""
    if request.app.state.startup_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The application dependencies are not ready.",
        )

    try:
        result = answer_question(
            question=payload.question,
            embedding_model=request.app.state.embedding_model,
            qdrant=request.app.state.qdrant,
            gemini=request.app.state.gemini,
        )

    except EvidenceInsufficientError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "status": "insufficient_evidence",
                "message": str(exc),
            },
        ) from exc

    except Exception:
        logger.exception(
            "question_answering_failed",
            extra={
                "request_id": getattr(
                    request.state,
                    "request_id",
                    "unknown",
                ),
                "method": request.method,
                "path": request.url.path,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "An unexpected error occurred while "
                "processing the question."
            ),
        ) from None

    return AskResponse(
        status="answered",
        answer=result["answer"],
        citations=result["citations"],
        validation_passed=result["validation"]["passed"],
    )
@app.get("/metrics", include_in_schema=False)
def metrics():
    """Expose metrics in Prometheus exposition format."""
    content, content_type = render_metrics()
    return Response(
        content=content,
        media_type=content_type.split(";")[0],
        headers={"Cache-Control": "no-store"},
    )