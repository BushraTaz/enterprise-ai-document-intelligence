
import re


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "did", "do",
    "does", "for", "from", "how", "in", "is", "it", "of",
    "on", "or", "the", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with", "according",
    "report", "group", "volvo",
}


def tokenize(text):
    """Convert text into lowercase words and numbers."""
    return re.findall(r"\b[a-z]+\b|\b\d[\d,]*(?:\.\d+)?\b", text.lower())


def assess_evidence_presence(question, evidence_texts, threshold=0.6):
    """
    Estimate whether retrieved text overlaps sufficiently with a question.

    This is a lexical screening heuristic, not proof that a question
    is answerable. It makes no API calls.

    Returns:
        answerability_signal: bool
        overlap_ratio: float
        matched_terms: sorted list of matched terms
        missing_terms: sorted list of missing terms
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be greater than 0 and at most 1")

    if not question or not question.strip():
        return {
            "answerability_signal": False,
            "overlap_ratio": 0.0,
            "matched_terms": [],
            "missing_terms": [],
            "reason": "The question is empty.",
        }

    if isinstance(evidence_texts, str):
        evidence_texts = [evidence_texts]

    evidence = " ".join(
        text for text in evidence_texts
        if isinstance(text, str)
    )

    if not evidence.strip():
        return {
            "answerability_signal": False,
            "overlap_ratio": 0.0,
            "matched_terms": [],
            "missing_terms": [],
            "reason": "No evidence text was provided.",
        }

    question_terms = {
        token
        for token in tokenize(question)
        if token not in STOP_WORDS
    }

    if not question_terms:
        return {
            "answerability_signal": False,
            "overlap_ratio": 0.0,
            "matched_terms": [],
            "missing_terms": [],
            "reason": "The question has no meaningful terms to compare.",
        }

    evidence_terms = set(tokenize(evidence))
    matched_terms = question_terms & evidence_terms
    missing_terms = question_terms - evidence_terms

    overlap_ratio = len(matched_terms) / len(question_terms)

    signal = overlap_ratio >= threshold

    if signal:
        reason = "Sufficient lexical overlap; factual support is not guaranteed."
    else:
        reason = "Insufficient lexical overlap between the question and evidence."

    return {
        "answerability_signal": signal,
        "overlap_ratio": round(overlap_ratio, 3),
        "matched_terms": sorted(matched_terms),
        "missing_terms": sorted(missing_terms),
        "reason": reason,
    }
