
import re


def has_page_citation(answer):
    """Check whether an answer contains a citation like [Page 8]."""
    return bool(
        re.search(r"\[Page\s+\d+\]", answer, re.IGNORECASE)
    )


def cited_pages(answer):
    """Return all page numbers cited in an answer."""
    return {
        int(page)
        for page in re.findall(
            r"\[Page\s+(\d+)\]",
            answer,
            re.IGNORECASE,
        )
    }


def cites_retrieved_page(answer, retrieved_pages):
    """Check whether at least one cited page was retrieved."""
    available_pages = {int(page) for page in retrieved_pages}
    return bool(cited_pages(answer) & available_pages)


def figures_in_text(text):
    """Extract numeric figures, including comma-separated numbers."""
    return set(
        re.findall(
            r"(?<!\w)\d[\d,]*(?:\.\d+)?",
            text,
        )
    )


def figures_supported_by_cited_evidence(answer, evidence_by_page):
    """
    Check whether numeric figures appear in evidence from cited pages.

    This is a basic heuristic, not a complete fact-checker.
    """
    pages = cited_pages(answer)

    if not pages:
        return False

    answer_without_citations = re.sub(
        r"\[Page\s+\d+\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    # Ignore numbered-list markers at the beginning of each line.
    answer_without_citations = re.sub(
        r"(?m)^\s*\d+\.\s+",
        "",
        answer_without_citations,
    )

    answer_figures = figures_in_text(answer_without_citations)

    if not answer_figures:
        return True

    evidence_figures = set()

    for page in pages:
        evidence = evidence_by_page.get(page, "")
        evidence_figures.update(figures_in_text(evidence))

    return answer_figures.issubset(evidence_figures)


def metrics_consistent_with_cited_evidence(answer, evidence_by_page):
    """
    Basic rule-based check for employee and net-sales metric mismatches.

    This catches a few known metric-label errors. It is not a
    general-purpose semantic fact-checker.
    """
    pages = cited_pages(answer)

    if not pages:
        return False

    answer_lower = answer.lower()

    evidence_parts = [
        evidence_by_page.get(page, "")
        for page in pages
    ]
    evidence_lower = " ".join(evidence_parts).lower()

    employee_terms = (
        "employee",
        "employees",
        "workforce",
        "staff",
    )
    sales_terms = (
        "net sales",
        "sales revenue",
    )

    answer_mentions_employees = any(
        term in answer_lower for term in employee_terms
    )
    answer_mentions_sales = any(
        term in answer_lower for term in sales_terms
    )

    evidence_mentions_employees = any(
        term in evidence_lower for term in employee_terms
    )
    evidence_mentions_sales = any(
        term in evidence_lower for term in sales_terms
    )

    # Reject an employee claim supported only by sales evidence.
    if (
        answer_mentions_employees
        and evidence_mentions_sales
        and not evidence_mentions_employees
    ):
        return False

    # Reject a sales claim supported only by employee evidence.
    if (
        answer_mentions_sales
        and evidence_mentions_employees
        and not evidence_mentions_sales
    ):
        return False

    return True


def validate_answer(answer, retrieved_pages, evidence_by_page):
    """
    Return validation results without modifying the answer.
    """
    citation_present = has_page_citation(answer)

    citation_matches_retrieval = cites_retrieved_page(
        answer,
        retrieved_pages,
    )

    figures_supported = figures_supported_by_cited_evidence(
        answer,
        evidence_by_page,
    )
    metric_consistent = metrics_consistent_with_cited_evidence(
    answer,
    evidence_by_page,
    )
    return {
        "citation_present": citation_present,
        "citation_matches_retrieval": citation_matches_retrieval,
        "figures_supported_by_cited_evidence": figures_supported,
        "metrics_consistent": metric_consistent,
        "passed": (
            citation_present
            and citation_matches_retrieval
            and figures_supported
            and metric_consistent
        ),
    }
