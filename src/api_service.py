
"""Service layer connecting the FastAPI application to the RAG pipeline."""

from src.rag import(
    build_prompt,
    check_retrieved_evidence,
    retrieve_chunks,
    validate_generated_answer,
)


class EvidenceInsufficientError(Exception):
    """Raised when retrieved evidence cannot safely support an answer."""


def answer_question(question, embedding_model, qdrant, gemini):
    """
    Answer a question using the existing RAG pipeline.

    External dependencies are injected so this function can be tested
    with mocks and without making real Gemini API requests.
    """
    results = retrieve_chunks(
        question,
        embedding_model,
        qdrant,
        limit=5,
    )

    if not results:
        raise EvidenceInsufficientError(
            "No relevant sources were retrieved."
        )

    gate_report = check_retrieved_evidence(question, results)

    if gate_report["decision"] != "ALLOW":
        raise EvidenceInsufficientError(
            gate_report["reason"]
        )

    response = gemini.interactions.create(
        model="gemini-3.8-flash",
        input=build_prompt(question, results),
    )

    answer = response.output_text
    validation = validate_generated_answer(answer, results)

    citations = sorted(
        {
            int(page)
            for result in results
            if (page := (result.payload or {}).get("page")) is not None
            and str(page).isdigit()
            and f"[Page {page}]" in answer
        }
    )

    return {
        "answer": answer,
        "citations": [{"page": page} for page in citations],
        "validation": validation,
        "gate_decision": gate_report["decision"],
    }
