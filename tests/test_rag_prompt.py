
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag import build_prompt


def make_result(page, text, source="volvo_annual_report.pdf"):
    return SimpleNamespace(
        payload={
            "page": page,
            "text": text,
            "source": source,
        }
    )


def test_prompt_includes_question():
    prompt = build_prompt(
        "What were Volvo Group's net sales in 2025?",
        [make_result(8, "Net sales, SEK M 479,183")],
    )

    assert "What were Volvo Group's net sales in 2025?" in prompt


def test_prompt_includes_page_and_evidence():
    prompt = build_prompt(
        "What were net sales?",
        [make_result(8, "Net sales, SEK M 479,183")],
    )

    assert "Page: 8" in prompt
    assert "Net sales, SEK M 479,183" in prompt


def test_prompt_includes_source_name():
    prompt = build_prompt(
        "What were net sales?",
        [make_result(8, "Net sales, SEK M 479,183")],
    )

    assert "volvo_annual_report.pdf" in prompt


def test_prompt_requires_citations_and_honest_answers():
    prompt = build_prompt(
        "What were net sales?",
        [make_result(8, "Net sales, SEK M 479,183")],
    )

    assert "[Page 8]" in prompt
    assert "Do not invent facts" in prompt
    assert "evidence is insufficient" in prompt.lower()


def test_prompt_preserves_employee_terminology():
    prompt = build_prompt(
        "How many employees did Volvo Group have?",
        [
            make_result(
                127,
                "Average number of employees: 89,188",
            )
        ],
    )

    assert "Average number of employees: 89,188" in prompt
    assert "total number of employees" in prompt
