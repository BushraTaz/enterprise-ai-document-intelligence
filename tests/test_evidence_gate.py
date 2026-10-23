
from evidence_gate import evaluate_evidence


def test_allows_matching_net_sales_evidence():
    result = evaluate_evidence(
        "What were Volvo Group net sales in 2025?",
        [
            "Volvo Group net sales in 2025 were "
            "479,183 SEK M."
        ],
    )

    assert result["decision"] == "ALLOW"


def test_declines_when_evidence_is_missing():
    result = evaluate_evidence(
        "What were net sales in 2025?",
        [],
    )

    assert result["decision"] == "DECLINE"


def test_declines_when_question_is_empty():
    result = evaluate_evidence(
        "   ",
        ["Volvo Group net sales were 479,183 SEK M in 2025."],
    )

    assert result["decision"] == "DECLINE"


def test_reviews_when_metric_does_not_match():
    result = evaluate_evidence(
        "What was the total number of employees in 2025?",
        [
            "The average number of employees in 2025 "
            "was 89,188."
        ],
    )

    assert result["decision"] == "REVIEW"


def test_reviews_when_exact_date_is_missing():
    result = evaluate_evidence(
        "What was daily employee attendance on 15 March 2025?",
        [
            "Daily employee attendance information is "
            "not included in this summary for 2025."
        ],
        answerability_threshold=0.4,
    )

    assert result["decision"] == "REVIEW"


def test_result_contains_both_check_reports():
    result = evaluate_evidence(
        "What were Volvo Group net sales in 2025?",
        ["Net sales in 2025 were 479,183 SEK M."],
    )

    assert "answerability" in result
    assert "specificity" in result
    assert "reason" in result
    assert result["decision"] in {
        "ALLOW",
        "REVIEW",
        "DECLINE",
    }


def test_accepts_a_single_evidence_string():
    result = evaluate_evidence(
        "What were Volvo Group net sales in 2025?",
        "Net sales in 2025 were 479,183 SEK M.",
    )

    assert result["decision"] == "ALLOW"
