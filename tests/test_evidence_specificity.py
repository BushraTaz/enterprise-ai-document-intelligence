
from evidence_specificity import assess_evidence_specificity


def test_matching_total_employee_metric_and_year():
    result = assess_evidence_specificity(
        "What was the total number of employees in 2025?",
        ["The total number of employees was 98,844 in 2025."],
    )

    assert result["specificity_signal"] is True
    assert result["metric_match"] is True
    assert result["date_match"] is True


def test_rejects_average_employee_metric_for_total_employee_question():
    result = assess_evidence_specificity(
        "What was the total number of employees in 2025?",
        ["The average number of employees was 89,188 in 2025."],
    )

    assert result["specificity_signal"] is False
    assert result["metric_match"] is False


def test_rejects_annual_data_for_exact_daily_attendance_date():
    result = assess_evidence_specificity(
        "What was employee attendance on 15 March 2025?",
        ["Total employees were 98,844 in 2025."],
    )

    assert result["specificity_signal"] is False
    assert result["date_match"] is False


def test_matching_net_sales_metric_and_year():
    result = assess_evidence_specificity(
        "What were net sales in 2025?",
        ["Net sales were SEK M 479,183 in 2025."],
    )

    assert result["specificity_signal"] is True


def test_rejects_wrong_year():
    result = assess_evidence_specificity(
        "What were net sales in 2025?",
        ["Net sales were SEK M 526,816 in 2024."],
    )

    assert result["specificity_signal"] is False
    assert result["date_match"] is False


def test_empty_evidence_is_not_sufficient():
    result = assess_evidence_specificity(
        "What were net sales in 2025?",
        [],
    )

    assert result["specificity_signal"] is False


def test_unrecognized_metric_does_not_automatically_fail():
    result = assess_evidence_specificity(
        "When was Volvo Group founded?",
        ["Volvo Group was founded in 1927."],
    )

    assert result["specificity_signal"] is True
    assert result["question_metric"] is None
