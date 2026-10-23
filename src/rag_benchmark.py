
import json
import os
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    retrieve_chunks,
    build_prompt,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "rag_benchmark_results.jsonl"
USAGE_FILE = RESULTS_DIR / "gemini_usage.json"
QDRANT_URL = "http://localhost:6333"


def now():
    return datetime.now(timezone.utc).isoformat()


def load_results():
    cached = {}
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get("status") == "success":
                        cached[item["cache_key"]] = item
                except (json.JSONDecodeError, KeyError):
                    continue
    return cached


def make_cache_key(question):
    value = f"{GEMINI_MODEL}|{question.strip()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_result(item):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_usage():
    if USAGE_FILE.exists():
        try:
            return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("Warning: usage log is unreadable; starting a new local log.")
    return {"total_attempts": 0, "by_model": {}, "events": []}


def record_attempt(usage):
    # Record before calling the API, so failed attempts are tracked too.
    usage["total_attempts"] += 1
    usage["by_model"][GEMINI_MODEL] = (
        usage["by_model"].get(GEMINI_MODEL, 0) + 1
    )
    usage["events"].append({
        "timestamp": now(),
        "model": GEMINI_MODEL,
        "type": "generation_attempt",
    })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    USAGE_FILE.write_text(
        json.dumps(usage, indent=2),
        encoding="utf-8",
    )


def main():
    load_dotenv()

    question = input("Enter your Volvo annual report question: ").strip()
    if not question:
        print("No question entered.")
        return

    cache_key = make_cache_key(question)
    cached = load_results().get(cache_key)

    if cached:
        print("\n[CACHED RESULT — no Gemini request made]")
        print("Model:", cached["model"])
        print("Question:", cached["question"])
        print("Retrieved pages:", cached.get("pages", []))
        print("\n--- SAVED ANSWER ---")
        print(cached.get("answer", ""))
        print("\nSaved at:", cached.get("timestamp"))
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    print("\nLoading local embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    qdrant = QdrantClient(url=QDRANT_URL)

    start = perf_counter()
    results = retrieve_chunks(question, model, qdrant, limit=5)
    retrieval_seconds = perf_counter() - start

    pages = [
        result.payload.get("page")
        for result in results
        if result.payload
    ]
    prompt = build_prompt(question, results)

    print(f"Retrieved {len(results)} chunks in {retrieval_seconds:.3f}s")
    print("Retrieved pages:", pages)

    print("\nBefore continuing, verify the selected model is eligible")
    print("for free use on your account and billing remains disabled.")
    confirm = input("Type RUN_ONE_GEMINI_REQUEST to continue: ").strip()

    if confirm != "RUN_ONE_GEMINI_REQUEST":
        print("Cancelled. No Gemini generation request made.")
        return

    usage = load_usage()
    record_attempt(usage)

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )

    generation_start = perf_counter()

    try:
        response = client.interactions.create(
            model=GEMINI_MODEL,
            input=prompt,
        )
        generation_seconds = perf_counter() - generation_start
        answer = response.output_text or ""

        item = {
            "cache_key": cache_key,
            "timestamp": now(),
            "status": "success",
            "model": GEMINI_MODEL,
            "question": question,
            "pages": pages,
            "retrieved_chunks": len(results),
            "retrieval_seconds": round(retrieval_seconds, 4),
            "generation_seconds": round(generation_seconds, 4),
            "total_seconds": round(
                retrieval_seconds + generation_seconds, 4
            ),
            "answer": answer,
        }
        save_result(item)

        print("\n--- ANSWER ---")
        print(answer)
        print(f"\nRetrieval: {retrieval_seconds:.3f}s")
        print(f"Generation: {generation_seconds:.3f}s")
        print(f"Total measured: {item['total_seconds']:.3f}s")
        print(f"\nSaved result to: {RESULTS_FILE}")
        print(f"Updated usage log: {USAGE_FILE}")

    except Exception as exc:
        item = {
            "cache_key": cache_key,
            "timestamp": now(),
            "status": "error",
            "model": GEMINI_MODEL,
            "question": question,
            "pages": pages,
            "retrieval_seconds": round(retrieval_seconds, 4),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        save_result(item)
        print("\nRequest failed:", type(exc).__name__)
        print("Failure recorded locally. No application-level retry.")


if __name__ == "__main__":
    main()