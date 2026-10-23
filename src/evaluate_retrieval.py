
import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "volvo_annual_report"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 10
QDRANT_URL = "http://localhost:6333"

QUESTIONS_PATH = Path("tests/rag_questions.json")
OUTPUT_PATH = Path("reports/retrieval_evaluation.json")


def load_questions():
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("The evaluation dataset must be a JSON list.")

    if not questions:
        raise ValueError("No evaluation questions found.")

    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Question {index} must be a JSON object."
            )

        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(
                f"Question {index} must contain a non-empty question."
            )

        # Backward compatibility: older datasets without "answerable"
        # are treated as answerable.
        answerable = item.get("answerable", True)

        if not isinstance(answerable, bool):
            raise ValueError(
                f"Question {index}: 'answerable' must be true or false."
            )

        if answerable and "expected_page" not in item:
            raise ValueError(
                f"Question {index} is answerable but has no expected_page."
            )

    return questions


def retrieve_one(query, model, client):
    vector = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=TOP_K,
        with_payload=True,
    )

    return response.points


def retrieve(question, model, client):
    queries = [question]
    normalized_question = question.lower()

    employee_question = any(
        word in normalized_question
        for word in ("employee", "employees", "workforce", "staff")
    )

    # Existing employee query expansion.
    if employee_question:
        queries.append(
            "Volvo Group number of employees in 2025 and 2024 "
            "89,188 90,995 employee count"
        )
        queries.append(
            "Volvo Group key ratios total number of employees "
            "2025 98,844 2024 101,595"
        )

    # Existing key-ratios employee query expansion.
    if employee_question and "key ratios" in normalized_question:
        queries.append(
            "Volvo Group Key ratios total number of employees "
            "2025 98,844 2024 101,595 page 8"
        )

    # Existing key-ratios net-sales query expansion.
    if (
        "net sales" in normalized_question
        and "key ratios" in normalized_question
    ):
        queries.append(
            "Volvo Group Key ratios 2025 2024 "
            "Net sales SEK M 479,183 526,816 page 8"
        )

    merged = {}

    for query in queries:
        for result in retrieve_one(query, model, client):
            key = str(result.id)

            if key not in merged or result.score > merged[key].score:
                merged[key] = result

    return sorted(
        merged.values(),
        key=lambda result: result.score,
        reverse=True,
    )[:TOP_K]


def get_ranked_pages(results):
    """Return distinct page numbers in ranked retrieval order."""
    pages = []

    for result in results:
        payload = result.payload or {}
        page = payload.get("page")

        if page is not None:
            normalized_page = str(page)

            if normalized_page not in [str(existing) for existing in pages]:
                pages.append(page)

    return pages


def evaluate_question(item, model, client):
    question = item["question"]
    answerable = item.get("answerable", True)

    results = retrieve(question, model, client)
    pages = get_ranked_pages(results)

    if not answerable:
        # Retrieval alone cannot determine whether the application
        # correctly abstains. Do not assign an expected page or score.
        result = {
            "question": question,
            "answerable": False,
            "expected_page": None,
            "rank": None,
            "reciprocal_rank_at_10": None,
            "hit_at_5": None,
            "hit_at_10": None,
            "top_5_pages": pages[:5],
            "top_10_pages": pages[:10],
            "abstention_evaluation": "NOT_EVALUATED",
        }

        print(question)
        print("  Expected answer: UNANSWERABLE")
        print(f"  Retrieved pages: {pages[:5]}")
        print("  Retrieval metrics: EXCLUDED")
        print("  Abstention correctness: NOT EVALUATED")
        print()

        return result

    expected_page = item["expected_page"]
    normalized_pages = [str(page) for page in pages]
    expected_page_normalized = str(expected_page)

    rank = (
        normalized_pages.index(expected_page_normalized) + 1
        if expected_page_normalized in normalized_pages
        else None
    )

    hit_at_5 = rank is not None and rank <= 5
    hit_at_10 = rank is not None and rank <= 10
    reciprocal_rank = 1.0 / rank if hit_at_10 else 0.0

    result = {
        "question": question,
        "answerable": True,
        "expected_page": expected_page,
        "rank": rank,
        "reciprocal_rank_at_10": reciprocal_rank,
        "hit_at_5": hit_at_5,
        "hit_at_10": hit_at_10,
        "top_5_pages": pages[:5],
        "top_10_pages": pages[:10],
    }

    print(question)
    print(f"  Expected page: {expected_page}")
    print(f"  Top 5 distinct pages:  {pages[:5]}")
    print(f"  Top 10 distinct pages: {pages[:10]}")
    print(f"  Expected-page hit@5:  {'PASS' if hit_at_5 else 'FAIL'}")
    print(f"  Expected-page hit@10: {'PASS' if hit_at_10 else 'FAIL'}")
    print(
        f"  Expected-page rank: "
        f"{rank if rank is not None else 'NOT FOUND'}"
    )
    print(f"  Reciprocal rank@10: {reciprocal_rank:.3f}")
    print()

    return result


def calculate_summary(results):
    answerable_results = [
        result for result in results
        if result["answerable"] is True
    ]

    unanswerable_results = [
        result for result in results
        if result["answerable"] is False
    ]

    answerable_count = len(answerable_results)

    if answerable_count:
        hit_at_5_count = sum(
            result["hit_at_5"] for result in answerable_results
        )
        hit_at_10_count = sum(
            result["hit_at_10"] for result in answerable_results
        )

        hit_rate_at_5 = hit_at_5_count / answerable_count
        hit_rate_at_10 = hit_at_10_count / answerable_count
        mrr_at_10 = sum(
            result["reciprocal_rank_at_10"]
            for result in answerable_results
        ) / answerable_count
    else:
        hit_at_5_count = 0
        hit_at_10_count = 0
        hit_rate_at_5 = None
        hit_rate_at_10 = None
        mrr_at_10 = None

    summary = {
        "evaluation_type": "expanded_query_retrieval",
        "total_question_count": len(results),
        "answerable_question_count": answerable_count,
        "unanswerable_question_count": len(unanswerable_results),
        "retrieval_metrics_denominator": answerable_count,
        "hit_at_5_count": hit_at_5_count,
        "hit_rate_at_5": hit_rate_at_5,
        "hit_at_10_count": hit_at_10_count,
        "hit_rate_at_10": hit_rate_at_10,
        "mrr_at_10": mrr_at_10,
        "abstention_evaluation": {
            "status": "NOT_EVALUATED",
            "questions_to_test": len(unanswerable_results),
            "reason": (
                "Retrieval metrics do not measure whether the answer "
                "generation service correctly abstains. Test the evidence "
                "gate or API response separately."
            ),
        },
    }

    return summary


def main():
    questions = load_questions()

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(url=QDRANT_URL)

    try:
        print("\n=== EXPANDED-QUERY RETRIEVAL EVALUATION ===\n")

        results = [
            evaluate_question(item, model, client)
            for item in questions
        ]

    finally:
        client.close()

    summary = calculate_summary(results)

    report = {
        "summary": summary,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("=== SUMMARY ===")
    print(
        f"Total questions: {summary['total_question_count']}"
    )
    print(
        f"Answerable questions: {summary['answerable_question_count']}"
    )
    print(
        f"Unanswerable questions: "
        f"{summary['unanswerable_question_count']}"
    )

    if summary["retrieval_metrics_denominator"]:
        denominator = summary["retrieval_metrics_denominator"]

        print(
            f"Hit@5: {summary['hit_at_5_count']}/{denominator} "
            f"({summary['hit_rate_at_5'] * 100:.1f}%)"
        )
        print(
            f"Hit@10: {summary['hit_at_10_count']}/{denominator} "
            f"({summary['hit_rate_at_10'] * 100:.1f}%)"
        )
        print(f"MRR@10: {summary['mrr_at_10']:.3f}")
    else:
        print("No answerable questions; retrieval metrics unavailable.")

    print("Abstention evaluation: NOT EVALUATED")
    print(f"\nDetailed report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
