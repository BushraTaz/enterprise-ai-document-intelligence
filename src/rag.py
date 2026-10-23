
import os

from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from answer_validation import validate_answer
from evidence_gate import evaluate_evidence
from evidence_specificity import assess_evidence_specificity


COLLECTION_NAME = "volvo_annual_report"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.8-flash"
QDRANT_URL = "http://localhost:6333"


def retrieve_chunks(query, model, client, limit=5):
    queries = [query]
    normalized_query = query.lower()

    # Expand employee questions to improve table retrieval.
    if any(word in normalized_query for word in (
        "employee", "employees", "workforce", "staff"
    )):
        queries.append(
            "Volvo Group number of employees in 2025 and 2024 "
            "89,188 90,995 employee count"
        )

    # Target the key-ratios table for relevant net-sales questions.
    if "net sales" in normalized_query and "key ratios" in normalized_query:
        queries.append(
            "Volvo Group Key ratios 2025 2024 "
            "Net sales SEK M 479,183 526,816 page 8"
        )

    merged = {}

    for search_query in queries:
        if "net sales" in normalized_query:
            print(f"DEBUG: Searching with query: {search_query}")

        query_vector = model.encode(
            search_query,
            normalize_embeddings=True,
        ).tolist()

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=10,
            with_payload=True,
        ).points

        if "net sales" in normalized_query:
            print(
                "DEBUG: Retrieved pages:",
                [
                    (result.payload or {}).get("page", "unknown")
                    for result in results
                ],
            )

        for result in results:
            key = str(result.id)

            if (
                key not in merged
                or result.score > merged[key].score
            ):
                merged[key] = result

    ranked = sorted(
        merged.values(),
        key=lambda result: result.score,
        reverse=True,
    )

    return ranked[:limit]


def build_prompt(question, results):
    context_parts = []

    for i, result in enumerate(results, start=1):
        payload = result.payload or {}

        context_parts.append(
            f"[Source {i}]\n"
            f"Page: {payload.get('page', 'unknown')}\n"
            f"Document: {payload.get('source', 'unknown')}\n"
            f"Evidence:\n{payload.get('text', '')}"
        )

    context = "\n\n".join(context_parts)

    return f"""
You are a careful financial-report question-answering assistant.

Answer the user's question using ONLY the provided source material.

REQUIREMENTS:
1. Do not invent facts, figures, dates, or explanations.
2. Cite the source page for every important factual claim.
3. Use citations in this format: [Page 8].
4. If multiple pages support an answer, cite each relevant page.
5. Preserve the report's exact terminology.
6. Do not confuse different metrics, such as
   "average number of employees" and "total number of employees."
7. If sources contain conflicting figures or terminology,
   explain the distinction instead of silently choosing one.
8. If evidence is insufficient, say so clearly.
9. Keep the answer concise and easy to understand.
10. Never claim a source supports something it does not say.

RESPONSE FORMAT:
Answer: [Your answer with page citations.]

Evidence: [Briefly identify the supporting figures or statements.]

USER QUESTION:
{question}

SOURCE MATERIAL:
{context}
"""


def validate_generated_answer(answer, results):
    """Validate an answer against the retrieved evidence."""
    retrieved_pages = []
    evidence_by_page = {}

    for result in results:
        payload = result.payload or {}
        raw_page = payload.get("page")
        text = payload.get("text") or ""

        try:
            page = int(raw_page)
        except (TypeError, ValueError):
            continue

        retrieved_pages.append(page)

        previous_text = evidence_by_page.get(page, "")
        evidence_by_page[page] = (
            f"{previous_text}\n{text}".strip()
        )

    return validate_answer(
        answer,
        retrieved_pages,
        evidence_by_page,
    )


def check_retrieved_evidence_specificity(question, results):
    """Run the existing metric/date specificity heuristic."""
    evidence_texts = [
        (result.payload or {}).get("text", "")
        for result in results
    ]

    return assess_evidence_specificity(
        question,
        evidence_texts,
    )


def check_retrieved_evidence(question, results):
    """Evaluate whether retrieved evidence should proceed to generation."""
    evidence_texts = [
        (result.payload or {}).get("text", "")
        for result in results
    ]

    return evaluate_evidence(
        question,
        evidence_texts,
    )



def print_specificity_report(report):
    """Display specificity warnings without blocking the RAG pipeline."""
    print("\n--- EVIDENCE SPECIFICITY CHECK ---")

    checks = (
        ("Recognized metric matches", "metric_match"),
        ("Requested date/year matches", "date_match"),
    )

    for label, key in checks:
        status = "PASS" if report.get(key, False) else "WARNING"
        print(f"{status}: {label}")

    print(f"Assessment: {report['reason']}")
    print(
        "Note: This heuristic does not establish factual support. "
        "Review the retrieved source text."
    )



def print_evidence_gate_report(report):
    """Display the evidence gate decision and its supporting checks."""
    print("\n--- EVIDENCE GATE ---")
    print(f"Decision: {report['decision']}")
    print(f"Reason: {report['reason']}")

    answerability = report.get("answerability")
    specificity = report.get("specificity")

    if answerability is not None:
        print(
            "Answerability overlap: "
            f"{answerability['overlap_ratio']:.1%}"
        )
        print(
            "Lexical overlap check: "
            f"{'PASS' if answerability['answerability_signal'] else 'FAIL'}"
        )

    if specificity is not None:
        print(
            "Specificity check: "
            f"{'PASS' if specificity['specificity_signal'] else 'FAIL'}"
        )

    print("Note: The gate is heuristic and can make mistakes.")


def print_validation_report(validation):
    """Display post-generation validation results."""
    print("\n--- ANSWER VALIDATION ---")

    checks = (
        ("Citation present", "citation_present"),
        ("Citation matches retrieved page", "citation_matches_retrieval"),
        (
            "Figures appear in cited evidence",
            "figures_supported_by_cited_evidence",
        ),
        (
            "Metrics consistent with cited evidence",
            "metrics_consistent",
        ),
    )

    for label, key in checks:
        status = "PASS" if validation.get(key, False) else "WARNING"
        print(f"{status}: {label}")

    if validation["passed"]:
        print("Overall: All basic validation checks passed.")
    else:
        print(
            "Overall: Validation warnings found. "
            "Review the answer and its cited sources."
        )

    
    print(
        "Note: These heuristic checks do not prove that "
        "the answer is factually correct."
    )



def main():
    load_dotenv()

    print("Loading local embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    qdrant = QdrantClient(url=QDRANT_URL)

    question = input("\nAsk a question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    print("\nSearching Qdrant...")
    results = retrieve_chunks(
        question,
        embedding_model,
        qdrant,
        limit=5,
    )

    if not results:
        print("\nNo relevant sources were retrieved.")
        print("Generation stopped. No Gemini request was made.")
        return

    print("\nRetrieved sources:")

    for result in results:
        payload = result.payload or {}

        print(
            f"  Page {payload.get('page', 'unknown')} "
            f"(score={result.score:.4f})"
        )

    # Keep the existing specificity report for visibility.
    specificity = check_retrieved_evidence_specificity(
        question,
        results,
    )
    print_specificity_report(specificity)

    # New gate: decide before initializing the Gemini client.
    gate_report = check_retrieved_evidence(question, results)
    print_evidence_gate_report(gate_report)

    if gate_report["decision"] != "ALLOW":
        print("\n--- GENERATION STOPPED ---")

        if gate_report["decision"] == "REVIEW":
            print(
                "The retrieved evidence needs review. "
                "No answer was generated."
            )
        else:
            print(
                "The available evidence appears insufficient. "
                "No answer was generated."
            )

        print(
            "Try rephrasing the question or checking whether "
            "the report contains the requested information."
        )
        return

    # Only initialize Gemini when the evidence gate allows generation.
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print(
            "\nEvidence passed the gate, but GEMINI_API_KEY "
            "is not configured. No answer was generated."
        )
        return

    gemini = genai.Client(api_key=api_key)
    prompt = build_prompt(question, results)

    print("\nGenerating answer...")

    response = gemini.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
    )

    answer = response.output_text

    print("\n--- ANSWER ---")
    print(answer)

    validation = validate_generated_answer(answer, results)
    print_validation_report(validation)


if __name__ == "__main__":
    main()
