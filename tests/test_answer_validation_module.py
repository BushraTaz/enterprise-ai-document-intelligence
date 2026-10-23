
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from answer_validation import (
    cited_pages,
    cites_retrieved_page,
    figures_supported_by_cited_evidence,
    has_page_citation,
    validate_answer,
)


def test_extracts_multiple_cited_pages():
    answer = "Sales details are reported here. [Page 8] [Page 37]"

    assert cited_pages(answer) == {8, 37}


def test_detects_page_citation():
    assert has_page_citation("Net sales were 479,183. [Page 8]")


def test_detects_missing_page_citation():
    assert not has_page_citation("Net sales were 479,183.")


def test_citation_matches_retrieved_page():
    answer = "Net sales were 479,183. [Page 8]"

    assert cites_retrieved_page(answer, [8, 37])


def test_citation_does_not_match_retrieved_pages():
    answer = "Net sales were 479,183. [Page 99]"

    assert not cites_retrieved_page(answer, [8, 37])


def test_accepts_supported_numeric_figure():
    answer = "Net sales were SEK 479,183 M. [Page 8]"
    evidence = {8: "Net sales, SEK M 479,183"}

    assert figures_supported_by_cited_evidence(answer, evidence)


def test_rejects_unsupported_numeric_figure():
    answer = "Net sales were SEK 999,999 M. [Page 8]"
    evidence = {8: "Net sales, SEK M 479,183"}

    assert not figures_supported_by_cited_evidence(answer, evidence)


def test_rejects_figure_found_only_on_uncited_page():
    answer = "Net sales were SEK 479,183 M. [Page 8]"
    evidence = {
        8: "Different financial metric",
        37: "Net sales, SEK M 479,183",
    }

    assert not figures_supported_by_cited_evidence(answer, evidence)


def test_validation_returns_individual_results():
    answer = "Net sales were SEK 479,183 M. [Page 8]"
    evidence = {8: "Net sales, SEK M 479,183"}

    result = validate_answer(answer, [8, 37], evidence)

    assert result["citation_present"] is True
    assert result["citation_matches_retrieval"] is True
    assert result["figures_supported_by_cited_evidence"] is True
    assert result["passed"] is True


def test_validation_fails_when_citation_is_missing():
    answer = "Net sales were SEK 479,183 M."
    evidence = {8: "Net sales, SEK M 479,183"}

    result = validate_answer(answer, [8], evidence)

    assert result["citation_present"] is False
    assert result["passed"] is False


def test_numbered_list_items_are_not_treated_as_unsupported_figures():
    answer = (
        "1. Net sales were SEK 479,183 M. [Page 8]\n"
        "2. The report provides the figure. [Page 8]"
    )

    evidence_by_page = {
        8: "Net sales, SEK M 479,183"
    }

    validation = validate_answer(
        answer,
        [8],
        evidence_by_page,
    )

    assert validation["citation_present"] is True
    assert validation["citation_matches_retrieval"] is True
    assert validation["figures_supported_by_cited_evidence"] is True
    assert validation["passed"] is True


def test_detects_number_used_with_wrong_metric():
    answer = "The number of employees was 479,183. [Page 8]"

    evidence_by_page = {
        8: "Net sales, SEK M 479,183"
    }

    validation = validate_answer(
        answer,
        [8],
        evidence_by_page,
    )

    assert validation["passed"] is False


def test_rejects_answer_citing_page_not_retrieved():
    answer = "Net sales were SEK 479,183 M. [Page 99]"
    evidence_by_page = {
        99: "Net sales, SEK M 479,183"
    }

    validation = validate_answer(
        answer,
        [8],
        evidence_by_page,
    )

    assert validation["citation_present"] is True
    assert validation["citation_matches_retrieval"] is False
    assert validation["passed"] is False


def test_rejects_numeric_claim_without_supporting_cited_evidence():
    answer = "Daily attendance was 125 people. [Page 8]"
    evidence_by_page = {
        8: "The report discusses employee information."
    }

    validation = validate_answer(
        answer,
        [8],
        evidence_by_page,
    )

    assert validation["citation_matches_retrieval"] is True
    assert validation["figures_supported_by_cited_evidence"] is False
    assert validation["passed"] is False


def test_rejects_numeric_claim_when_no_evidence_is_available():
    answer = "Daily attendance was 125 people. [Page 8]"
    evidence_by_page = {}

    validation = validate_answer(
        answer,
        [8],
        evidence_by_page,
    )

    assert validation["figures_supported_by_cited_evidence"] is False
    assert validation["passed"] is False
