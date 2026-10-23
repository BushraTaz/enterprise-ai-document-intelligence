
from answerability import assess_evidence_presence


def test_relevant_evidence_has_sufficient_lexical_overlap():
    question = (
        "What was Volvo Group's total number of employees in 2025?"
    )
    evidence = [
        "Key ratios: total number of employees 98,844 in 2025."
    ]

    result = assess_evidence_presence(question, evidence)

    assert result["answerability_signal"] is True
    assert "employees" in result["matched_terms"]
    assert result["overlap_ratio"] >= 0.6


def test_unrelated_evidence_has_insufficient_overlap():
    question = (
        "What was Volvo Group's exact daily employee attendance "
        "on 15 March 2025?"
    )
    evidence = [
        "Net sales were SEK 479,183 M in 2025."
    ]

    result = assess_evidence_presence(question, evidence)

    assert result["answerability_signal"] is False
    assert result["overlap_ratio"] < 0.6


def test_empty_evidence_returns_false():
    result = assess_evidence_presence(
        "What were net sales in 2025?",
        [],
    )

    assert result["answerability_signal"] is False
    assert result["overlap_ratio"] == 0.0


def test_empty_question_returns_false():
    result = assess_evidence_presence("  ", ["Some report text."])

    assert result["answerability_signal"] is False


def test_multiple_evidence_chunks_are_combined():
    result = assess_evidence_presence(
        "What were net sales in 2025?",
        [
            "Net sales were reported in the annual report.",
            "The value was 479,183 SEK M in 2025.",
        ],
    )

    assert result["answerability_signal"] is True
    assert "sales" in result["matched_terms"]
    assert "2025" in result["matched_terms"]


def test_invalid_threshold_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        assess_evidence_presence(
            "What were net sales?",
            ["Net sales were 479,183."],
            threshold=0,
        )


def test_related_employee_evidence_does_not_prove_daily_attendance():
    question = (
        "What was Volvo Group's exact daily employee attendance "
        "on 15 March 2025?"
    )
    evidence = [
        "The key ratios table reports total employees in 2025 "
        "as 98,844. The overview says Volvo Group employs "
        "approximately 99,000 people."
    ]

    result = assess_evidence_presence(question, evidence)

    # This evidence does not establish the exact daily attendance.
    # Test the configured threshold, not specific matched words.
    assert result["answerability_signal"] is False
    assert result["overlap_ratio"] < 0.6
