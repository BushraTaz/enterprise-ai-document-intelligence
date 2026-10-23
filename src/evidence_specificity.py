
import re


def normalize_text(text):
    """Normalize text for simple, case-insensitive comparisons."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_years(text):
    """Extract four-digit years."""
    return set(re.findall(r"\b(?:19|20)\d{2}\b", text))


def extract_dates(text):
    """Extract dates written as day Month year, e.g. 15 March 2025."""
    pattern = (
        r"\b\d{1,2}\s+"
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+"
        r"(?:19|20)\d{2}\b"
    )
    return {
        normalize_text(date)
        for date in re.findall(pattern, text, flags=re.IGNORECASE)
    }


def identify_metric(text):
    """
    Identify a few report metrics that must not be confused.

    Returns None when no supported metric category is detected.
    """
    text = normalize_text(text)

    if "average number of employees" in text:
        return "average_employees"

    if (
        "total number of employees" in text
        or "total employees" in text
    ):
        return "total_employees"

    if "net sales" in text:
        return "net_sales"

    if "daily attendance" in text or "employee attendance" in text:
        return "daily_attendance"

    return None


def assess_evidence_specificity(question, evidence_texts):
    """
    Check whether retrieved evidence matches the question's
    detected metric and explicit dates or years.

    This is a heuristic screening check, not proof of answerability.
    It makes no API calls.
    """
    if not question or not question.strip():
        return {
            "specificity_signal": False,
            "metric_match": False,
            "date_match": False,
            "reason": "The question is empty.",
        }

    if isinstance(evidence_texts, str):
        evidence_texts = [evidence_texts]

    evidence = " ".join(
        item for item in evidence_texts
        if isinstance(item, str)
    )

    if not evidence.strip():
        return {
            "specificity_signal": False,
            "metric_match": False,
            "date_match": False,
            "reason": "No evidence text was provided.",
        }

    question_metric = identify_metric(question)
    evidence_metric = identify_metric(evidence)

    # If the question uses a recognized metric, require the evidence
    # to identify that same metric.
    metric_match = (
        question_metric is None
        or evidence_metric == question_metric
    )

    question_dates = extract_dates(question)
    evidence_dates = extract_dates(evidence)

    question_years = extract_years(question)
    evidence_years = extract_years(evidence)

    # An exact date in a question requires that date in the evidence.
    exact_date_match = (
        not question_dates
        or bool(question_dates & evidence_dates)
    )

    # If the question names years but no full dates, require at least
    # one requested year to appear in the evidence.
    year_match = (
        not question_years
        or bool(question_years & evidence_years)
    )

    date_match = exact_date_match and year_match

    signal = metric_match and date_match

    if signal:
        reason = (
            "Recognized metric and date/year checks passed. "
            "The answer still requires factual verification."
        )
    else:
        reason = (
            "The evidence may not match the requested metric "
            "or date/year."
        )

    return {
        "specificity_signal": signal,
        "metric_match": metric_match,
        "date_match": date_match,
        "question_metric": question_metric,
        "evidence_metric": evidence_metric,
        "question_dates": sorted(question_dates),
        "evidence_dates": sorted(evidence_dates),
        "question_years": sorted(question_years),
        "evidence_years": sorted(evidence_years),
        "reason": reason,
    }
