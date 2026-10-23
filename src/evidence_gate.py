
from answerability import assess_evidence_presence
from evidence_specificity import assess_evidence_specificity


def evaluate_evidence(
    question,
    evidence_texts,
    answerability_threshold=0.6,
):
    """
    Decide whether retrieved evidence should be allowed,
    reviewed, or declined before answer generation.

    This is a heuristic gate, not proof of factual correctness.
    It makes no external API calls.
    """
    if not question or not question.strip():
        return {
            "decision": "DECLINE",
            "answerability": None,
            "specificity": None,
            "reason": "The question is empty.",
        }

    if isinstance(evidence_texts, str):
        evidence_texts = [evidence_texts]

    evidence_texts = [
        text for text in evidence_texts
        if isinstance(text, str) and text.strip()
    ]

    if not evidence_texts:
        return {
            "decision": "DECLINE",
            "answerability": None,
            "specificity": None,
            "reason": "No usable evidence was retrieved.",
        }

    answerability = assess_evidence_presence(
        question,
        evidence_texts,
        threshold=answerability_threshold,
    )

    specificity = assess_evidence_specificity(
        question,
        evidence_texts,
    )

    overlap_passed = answerability["answerability_signal"]
    specificity_passed = specificity["specificity_signal"]

    if overlap_passed and specificity_passed:
        decision = "ALLOW"
        reason = (
            "Evidence passed the lexical overlap and "
            "metric/date specificity checks."
        )
    elif not overlap_passed and not specificity_passed:
        decision = "DECLINE"
        reason = (
            "Evidence failed both checks and appears "
            "insufficient for answering the question."
        )
    else:
        decision = "REVIEW"
        reason = (
            "The checks disagree or one check failed. "
            "Review the retrieved evidence before answering."
        )

    return {
        "decision": decision,
        "answerability": answerability,
        "specificity": specificity,
        "reason": reason,
    }
