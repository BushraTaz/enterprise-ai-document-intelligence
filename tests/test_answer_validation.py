
import re


def has_page_citation(answer):
    """Check whether an answer contains a citation like [Page 8]."""
    return bool(re.search(r"\[Page\s+\d+\]", answer, re.IGNORECASE))


def cites_retrieved_page(answer, retrieved_pages):
    """Check whether at least one cited page was retrieved."""
    cited_pages = {
        int(page)
        for page in re.findall(
            r"\[Page\s+(\d+)\]",
            answer,
            re.IGNORECASE,
        )
    }

    return bool(cited_pages & {int(page) for page in retrieved_pages})


def cited_pages(answer):
    """Return the page numbers cited in an answer."""
    return {
        int(page)
        for page in re.findall(
            r"\[Page\s+(\d+)\]",
            answer,
            re.IGNORECASE,
        )
    }


def figures_in_text(text):
    """Extract numeric figures, allowing common thousands separators."""
    return set(
        re.findall(
            r"(?<!\w)\d[\d,]*(?:\.\d+)?",
            text,
        )
    )


def figures_supported_by_cited_evidence(answer, evidence_by_page):
    """
    Check whether figures in the answer appear in the evidence
    for at least one page cited by that answer.

    evidence_by_page example:
        {8: "Net sales, SEK M 479,183"}
    """
    pages = cited_pages(answer)

    if not pages:
        return False

    answer_figures = figures_in_text(
        re.sub(r"\[Page\s+\d+\]", "", answer, flags=re.IGNORECASE)
    )

    if not answer_figures:
        return True

    evidence_figures = set()

    for page in pages:
        evidence = evidence_by_page.get(page, "")
        evidence_figures.update(figures_in_text(evidence))

    return answer_figures.issubset(evidence_figures)


def test_accepts_valid_page_citation():
    answer = "Net sales were SEK 479,183 M. [Page 8]"

    assert has_page_citation(answer)


def test_rejects_answer_without_page_citation():
    answer = "Net sales were SEK 479,183 M."

    assert not has_page_citation(answer)


def test_accepts_citation_to_retrieved_page():
    answer = "Net sales were SEK 479,183 M. [Page 8]"

    assert cites_retrieved_page(answer, [8, 37, 51])


def test_rejects_citation_not_in_retrieved_pages():
    answer = "Net sales were SEK 479,183 M. [Page 99]"

    assert not cites_retrieved_page(answer, [8, 37, 51])


def test_accepts_multiple_citations_when_one_page_was_retrieved():
    answer = "Net sales were reported in the key ratios. [Page 8] [Page 37]"

    assert cites_retrieved_page(answer, [8, 51])


def test_accepts_figure_supported_by_cited_evidence():
    answer = "Net sales were SEK 479,183 M. [Page 8]"
    evidence = {
        8: "Net sales, SEK M 479,183",
    }

    assert figures_supported_by_cited_evidence(answer, evidence)


def test_rejects_unsupported_figure():
    answer = "Net sales were SEK 999,999 M. [Page 8]"
    evidence = {
        8: "Net sales, SEK M 479,183",
    }

    assert not figures_supported_by_cited_evidence(answer, evidence)


def test_rejects_figure_found_only_on_an_uncited_page():
    answer = "Net sales were SEK 479,183 M. [Page 8]"
    evidence = {
        8: "Revenue table for a different metric",
        37: "Net sales, SEK M 479,183",
    }

    assert not figures_supported_by_cited_evidence(answer, evidence)


def test_rejects_evidence_check_without_citations():
    answer = "Net sales were SEK 479,183 M."
    evidence = {
        8: "Net sales, SEK M 479,183",
    }

    assert not figures_supported_by_cited_evidence(answer, evidence)
