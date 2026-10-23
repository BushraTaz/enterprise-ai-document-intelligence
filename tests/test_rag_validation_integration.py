
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag import (
    check_retrieved_evidence_specificity,
    print_specificity_report,
    validate_generated_answer,
    print_validation_report,
)


def make_result(page, text):
    return SimpleNamespace(
        payload={
            "page": page,
            "text": text,
            "source": "volvo_annual_report.pdf",
        }
    )


def test_validates_answer_against_retrieved_evidence():
    answer = "Net sales were SEK 479,183 M. [Page 8]"
    results = [
        make_result(8, "Net sales, SEK M 479,183"),
        make_result(37, "Net sales, SEK M 479,183"),
    ]

    validation = validate_generated_answer(answer, results)

    assert validation["citation_present"] is True
    assert validation["citation_matches_retrieval"] is True
    assert validation["figures_supported_by_cited_evidence"] is True
    assert validation["passed"] is True


def test_flags_citation_to_unretrieved_page():
    answer = "Net sales were SEK 479,183 M. [Page 99]"
    results = [
        make_result(8, "Net sales, SEK M 479,183"),
    ]

    validation = validate_generated_answer(answer, results)

    assert validation["citation_present"] is True
    assert validation["citation_matches_retrieval"] is False
    assert validation["figures_supported_by_cited_evidence"] is False
    assert validation["passed"] is False


def test_flags_unsupported_figure():
    answer = "Net sales were SEK 999,999 M. [Page 8]"
    results = [
        make_result(8, "Net sales, SEK M 479,183"),
    ]

    validation = validate_generated_answer(answer, results)

    assert validation["figures_supported_by_cited_evidence"] is False
    assert validation["passed"] is False


def test_combines_evidence_from_same_page():
    answer = "Net sales were SEK 479,183 M. [Page 8]"
    results = [
        make_result(8, "Net sales, SEK M"),
        make_result(8, "479,183"),
    ]

    validation = validate_generated_answer(answer, results)

    assert validation["figures_supported_by_cited_evidence"] is True


def test_ignores_results_with_unknown_page_numbers():
    answer = "Net sales were SEK 479,183 M. [Page 8]"
    results = [
        make_result("unknown", "Unnumbered source"),
        make_result(8, "Net sales, SEK M 479,183"),
    ]

    validation = validate_generated_answer(answer, results)

    assert validation["passed"] is True


def test_prints_validation_report(capsys):
    validation = {
        "citation_present": True,
        "citation_matches_retrieval": True,
        "figures_supported_by_cited_evidence": False,
        "metrics_consistent": False,
        "passed": False,
    }

    print_validation_report(validation)
    output = capsys.readouterr().out

    assert "PASS: Citation present" in output
    assert "WARNING: Figures appear in cited evidence" in output
    assert "WARNING: Metrics consistent with cited evidence" in output
    assert "Review the answer" in output
    assert "do not prove that" in output


def test_specificity_check_accepts_matching_net_sales_evidence():
    results = [
        make_result(8, "Net sales, SEK M 479,183 in 2025"),
    ]

    report = check_retrieved_evidence_specificity(
        "What were net sales in 2025?",
        results,
    )

    assert report["specificity_signal"] is True
    assert report["metric_match"] is True
    assert report["date_match"] is True


def test_specificity_check_flags_wrong_employee_metric():
    results = [
        make_result(
            127,
            "Average number of employees was 89,188 in 2025.",
        ),
    ]

    report = check_retrieved_evidence_specificity(
        "What was the total number of employees in 2025?",
        results,
    )

    assert report["specificity_signal"] is False
    assert report["metric_match"] is False


def test_specificity_check_flags_missing_exact_date():
    results = [
        make_result(
            8,
            "The total number of employees was 98,844 in 2025.",
        ),
    ]

    report = check_retrieved_evidence_specificity(
        "What was employee attendance on 15 March 2025?",
        results,
    )

    assert report["specificity_signal"] is False
    assert report["date_match"] is False


def test_prints_specificity_warning_report(capsys):
    report = {
        "specificity_signal": False,
        "metric_match": False,
        "date_match": True,
        "reason": "The evidence may use the wrong metric.",
    }

    print_specificity_report(report)
    output = capsys.readouterr().out

    assert "WARNING: Recognized metric matches" in output
    assert "PASS: Requested date/year matches" in output
    assert "wrong metric" in output
    assert "does not establish factual support" in output
