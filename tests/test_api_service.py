
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import api_service
from api_service import EvidenceInsufficientError, answer_question


@pytest.fixture
def dependencies():
    """Create fake dependencies without external service calls."""
    embedding_model = Mock(name="embedding_model")
    qdrant = Mock(name="qdrant")
    gemini = Mock(name="gemini")

    results = [
        SimpleNamespace(
            id="point-1",
            score=0.95,
            payload={
                "page": 8,
                "text": (
                    "Total number of employees was 98,844 in 2025."
                ),
            },
        ),
        SimpleNamespace(
            id="point-2",
            score=0.90,
            payload={
                "page": 127,
                "text": (
                    "Average number of employees was 89,188 in 2025."
                ),
            },
        ),
    ]

    return {
        "embedding_model": embedding_model,
        "qdrant": qdrant,
        "gemini": gemini,
        "results": results,
    }


def test_successful_answer_returns_citations_and_validation(
    monkeypatch,
    dependencies,
):
    """A permitted answer returns its answer, citations, and validation."""
    monkeypatch.setattr(
        api_service,
        "retrieve_chunks",
        Mock(return_value=dependencies["results"]),
    )

    monkeypatch.setattr(
        api_service,
        "check_retrieved_evidence",
        Mock(return_value={
            "decision": "ALLOW",
            "reason": "Evidence passed the checks.",
        }),
    )

    monkeypatch.setattr(
        api_service,
        "build_prompt",
        Mock(return_value="Test prompt"),
    )

    monkeypatch.setattr(
        api_service,
        "validate_generated_answer",
        Mock(return_value={
            "passed": True,
            "citation_present": True,
        }),
    )

    dependencies["gemini"].interactions.create.return_value = (
        SimpleNamespace(
            output_text=(
                "The total number of employees was 98,844 "
                "in 2025 [Page 8]."
            )
        )
    )

    result = answer_question(
        question="How many employees were there in 2025?",
        embedding_model=dependencies["embedding_model"],
        qdrant=dependencies["qdrant"],
        gemini=dependencies["gemini"],
    )

    assert result["answer"].startswith(
        "The total number of employees was 98,844"
    )
    assert result["citations"] == [{"page": 8}]
    assert result["validation"]["passed"] is True
    assert result["gate_decision"] == "ALLOW"

    dependencies["gemini"].interactions.create.assert_called_once()


def test_empty_retrieval_raises_error_without_calling_gemini(
    monkeypatch,
    dependencies,
):
    """No retrieved sources means no generation."""
    monkeypatch.setattr(
        api_service,
        "retrieve_chunks",
        Mock(return_value=[]),
    )

    with pytest.raises(
        EvidenceInsufficientError,
        match="No relevant sources",
    ):
        answer_question(
            question="What were the employee figures?",
            embedding_model=dependencies["embedding_model"],
            qdrant=dependencies["qdrant"],
            gemini=dependencies["gemini"],
        )

    dependencies["gemini"].interactions.create.assert_not_called()


def test_rejected_evidence_blocks_gemini(
    monkeypatch,
    dependencies,
):
    """Evidence rejected by the gate must not reach generation."""
    monkeypatch.setattr(
        api_service,
        "retrieve_chunks",
        Mock(return_value=dependencies["results"]),
    )

    monkeypatch.setattr(
        api_service,
        "check_retrieved_evidence",
        Mock(return_value={
            "decision": "DECLINE",
            "reason": "Evidence is insufficient.",
        }),
    )

    with pytest.raises(
        EvidenceInsufficientError,
        match="Evidence is insufficient",
    ):
        answer_question(
            question="What were the employee figures?",
            embedding_model=dependencies["embedding_model"],
            qdrant=dependencies["qdrant"],
            gemini=dependencies["gemini"],
        )

    dependencies["gemini"].interactions.create.assert_not_called()


def test_review_decision_also_blocks_gemini(
    monkeypatch,
    dependencies,
):
    """A REVIEW decision must not be treated as permission to generate."""
    monkeypatch.setattr(
        api_service,
        "retrieve_chunks",
        Mock(return_value=dependencies["results"]),
    )

    monkeypatch.setattr(
        api_service,
        "check_retrieved_evidence",
        Mock(return_value={
            "decision": "REVIEW",
            "reason": "Evidence checks disagree.",
        }),
    )

    with pytest.raises(
        EvidenceInsufficientError,
        match="Evidence checks disagree",
    ):
        answer_question(
            question="What were the employee figures?",
            embedding_model=dependencies["embedding_model"],
            qdrant=dependencies["qdrant"],
            gemini=dependencies["gemini"],
        )

    dependencies["gemini"].interactions.create.assert_not_called()


def test_answer_without_matching_page_citation_returns_no_citations(
    monkeypatch,
    dependencies,
):
    """Retrieved pages are not citations unless the answer cites them."""
    monkeypatch.setattr(
        api_service,
        "retrieve_chunks",
        Mock(return_value=dependencies["results"]),
    )

    monkeypatch.setattr(
        api_service,
        "check_retrieved_evidence",
        Mock(return_value={
            "decision": "ALLOW",
            "reason": "Evidence passed the checks.",
        }),
    )

    monkeypatch.setattr(
        api_service,
        "build_prompt",
        Mock(return_value="Test prompt"),
    )

    monkeypatch.setattr(
        api_service,
        "validate_generated_answer",
        Mock(return_value={
            "passed": False,
            "citation_present": False,
        }),
    )

    dependencies["gemini"].interactions.create.return_value = (
        SimpleNamespace(
            output_text="The employee figure is 98,844."
        )
    )

    result = answer_question(
        question="How many employees were there?",
        embedding_model=dependencies["embedding_model"],
        qdrant=dependencies["qdrant"],
        gemini=dependencies["gemini"],
    )

    assert result["citations"] == []
    assert result["validation"]["passed"] is False
    dependencies["gemini"].interactions.create.assert_called_once()


def test_unanswerable_daily_attendance_question_blocks_gemini(
    monkeypatch,
    dependencies,
):
    """An unsupported exact-date question must not reach generation."""
    monkeypatch.setattr(
        api_service,
        "retrieve_chunks",
        Mock(return_value=dependencies["results"]),
    )

    with pytest.raises(EvidenceInsufficientError):
        answer_question(
            question=(
                "What was Volvo Group's exact daily employee "
                "attendance on 15 March 2025?"
            ),
            embedding_model=dependencies["embedding_model"],
            qdrant=dependencies["qdrant"],
            gemini=dependencies["gemini"],
        )

    dependencies["gemini"].interactions.create.assert_not_called()
