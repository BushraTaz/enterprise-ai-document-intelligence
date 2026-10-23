
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag import retrieve_chunks


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "volvo_annual_report"
QUESTIONS_FILE = ROOT / "tests" / "rag_questions.json"
RESULTS_DIR = ROOT / "results"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "did",
    "do", "does", "for", "from", "how", "in", "is",
    "it", "of", "on", "or", "the", "to", "was", "were",
    "what", "when", "where", "which", "who", "with",
    "according", "report", "group",
}


def tokenize(text):
    return {
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9,]+", text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    }


def rerank_score(question, result):
    payload = result.payload or {}
    text = payload.get("text") or ""

    question_terms = tokenize(question)
    text_terms = tokenize(text)

    overlap = (
        len(question_terms & text_terms) / len(question_terms)
        if question_terms else 0.0
    )

    score = result.score + 0.08 * overlap

    if len(text.strip()) < 100:
        score -= 0.15

    return score


def page_of(result):
    return str((result.payload or {}).get("page", "unknown"))


def contains_answer(result, expected):
    text = (result.payload or {}).get("text") or ""
    return bool(expected) and expected.casefold() in text.casefold()


def first_matching_rank(results, predicate):
    for rank, result in enumerate(results, start=1):
        if predicate(result):
            return rank
    return None


def reciprocal_rank(rank):
    return 1.0 / rank if rank is not None else 0.0


def summarize(results_by_question, k, ranking, match_kind):
    hits = 0
    reciprocal_ranks = []
    evaluated = 0

    for item in results_by_question:
        expected_answer = item["expected_answer"]

        if match_kind == "answer" and not expected_answer:
            continue

        evaluated += 1
        results = item[ranking][:k]

        if match_kind == "answer":
            predicate = lambda result, answer=expected_answer: (
                contains_answer(result, answer)
            )
        else:
            expected_page = str(item["expected_page"])
            predicate = lambda result, page=expected_page: (
                page_of(result) == page
            )

        rank = first_matching_rank(results, predicate)

        if rank is not None:
            hits += 1

        reciprocal_ranks.append(reciprocal_rank(rank))

    return {
        "hits": hits,
        "total": evaluated,
        "recall": hits / evaluated if evaluated else 0.0,
        "mrr": (
            sum(reciprocal_ranks) / evaluated
            if evaluated else 0.0
        ),
    }


def print_metrics(label, metrics, k):
    print(
        f"{label} Recall@{k}: "
        f"{metrics['hits']}/{metrics['total']} "
        f"({metrics['recall']:.1%})"
    )
    print(f"{label} MRR@{k}: {metrics['mrr']:.3f}")


def build_report(results_by_question, question_count, run_id):
    metrics = {}

    for k in (5, 10):
        for ranking in ("baseline", "reranked"):
            for match_kind in ("page", "answer"):
                key = f"{ranking}_{match_kind}_at_{k}"
                metrics[key] = summarize(
                    results_by_question, k, ranking, match_kind
                )

    return {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "questions_file": str(QUESTIONS_FILE.relative_to(ROOT)),
            "question_count": question_count,
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "evaluation_mode": "local_retrieval_only",
        },
        "metrics": metrics,
        "limitations": [
            "The benchmark uses a small, manually curated question set.",
            "Answer-evidence matching checks exact expected text in retrieved chunks.",
            "Retrieval metrics do not prove generated answers are correct.",
            "Reranking only reorders retrieved candidates.",
        ],
    }


def save_report(report):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    historical_path = RESULTS_DIR / (
        f"retrieval_evaluation_{timestamp}.json"
    )
    latest_path = RESULTS_DIR / "retrieval_evaluation_latest.json"

    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"

    historical_path.write_text(serialized, encoding="utf-8")
    latest_path.write_text(serialized, encoding="utf-8")

    return historical_path, latest_path


def main():
    if not QUESTIONS_FILE.exists():
        print(f"Missing benchmark file: {QUESTIONS_FILE}")
        raise SystemExit(1)

    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        print("Benchmark has no questions.")
        raise SystemExit(1)

    print("Loading local embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(url=QDRANT_URL)

    results_by_question = []

    print(f"\nTesting {len(questions)} benchmark questions.")
    print("Local retrieval only. No Gemini requests will be made.\n")

    for item in questions:
        question = item["question"]
        expected_page = item["expected_page"]
        expected_answer = item.get("expected_answer_contains", "")

        results = retrieve_chunks(
            question,
            model,
            client,
            limit=10,
        )

        baseline = list(results)
        reranked = sorted(
            results,
            key=lambda result: rerank_score(question, result),
            reverse=True,
        )

        results_by_question.append({
            "question": question,
            "expected_page": expected_page,
            "expected_answer": expected_answer,
            "baseline": baseline,
            "reranked": reranked,
        })

        baseline_page_rank = first_matching_rank(
            baseline,
            lambda result: page_of(result) == str(expected_page),
        )
        reranked_page_rank = first_matching_rank(
            reranked,
            lambda result: page_of(result) == str(expected_page),
        )

        baseline_answer_rank = (
            first_matching_rank(
                baseline,
                lambda result: contains_answer(result, expected_answer),
            )
            if expected_answer else None
        )
        reranked_answer_rank = (
            first_matching_rank(
                reranked,
                lambda result: contains_answer(result, expected_answer),
            )
            if expected_answer else None
        )

        print("=" * 72)
        print(question)
        print(f"Expected page: {expected_page}")
        print(f"Expected answer text: {expected_answer or '(not specified)'}")
        print(
            f"Expected-page rank: baseline={baseline_page_rank}, "
            f"reranked={reranked_page_rank}"
        )
        print(
            f"Answer-evidence rank: baseline={baseline_answer_rank}, "
            f"reranked={reranked_answer_rank}"
        )

        if reranked_page_rank is None:
            print("Expected page missing from top 10.")
            for rank, result in enumerate(reranked, start=1):
                payload = result.payload or {}
                print(
                    f"  Rank {rank}: page={payload.get('page')}, "
                    f"score={result.score:.4f}"
                )

    print("\n" + "=" * 72)
    print("FINAL METRICS")
    print("=" * 72)

    for k in (5, 10):
        for ranking, ranking_label in (
            ("baseline", "Baseline"),
            ("reranked", "Reranked"),
        ):
            for match_kind, match_label in (
                ("page", "Page"),
                ("answer", "Answer-evidence"),
            ):
                metrics = summarize(
                    results_by_question, k, ranking, match_kind
                )
                print_metrics(f"{ranking_label} {match_label}", metrics, k)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = build_report(results_by_question, len(questions), run_id)
    historical_path, latest_path = save_report(report)

    print("\nJSON reports saved:")
    print(f"  Historical: {historical_path.relative_to(ROOT)}")
    print(f"  Latest:     {latest_path.relative_to(ROOT)}")

    print("\nEvaluation complete. No Gemini request was made.")
    print(
        "Page metrics check whether the expected page appears "
        "in retrieved results."
    )
    print(
        "Answer-evidence metrics check for exact expected text. "
        "They do not prove a generated answer is correct."
    )
    print(
        "Reranking only reorders retrieved candidates; "
        "it cannot recover evidence absent from the candidates."
    )


if __name__ == "__main__":
    main()
