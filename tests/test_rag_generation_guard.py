
import rag


class FakeResult:
    def __init__(self, page=8, text="Net sales in 2025 were 479,183 SEK M."):
        self.id = "test-point"
        self.score = 0.9
        self.payload = {
            "page": page,
            "text": text,
            "source": "test-report.pdf",
        }


class FakeEmbeddingModel:
    pass


class FakeQdrantClient:
    pass


class FakeGeminiResponse:
    output_text = (
        "Net sales were 479,183 SEK M in 2025. [Page 8]"
    )


class FakeGeminiInteractions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return FakeGeminiResponse()


class FakeGeminiClient:
    def __init__(self, api_key):
        self.interactions = FakeGeminiInteractions()


def configure_common_mocks(monkeypatch, decision):
    """Replace external dependencies so tests stay offline."""
    fake_gemini_clients = []

    monkeypatch.setattr(rag, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        rag,
        "SentenceTransformer",
        lambda model_name: FakeEmbeddingModel(),
    )
    monkeypatch.setattr(
        rag,
        "QdrantClient",
        lambda url: FakeQdrantClient(),
    )
    monkeypatch.setattr(
        rag,
        "retrieve_chunks",
        lambda *args, **kwargs: [FakeResult()],
    )
    monkeypatch.setattr(
        rag,
        "check_retrieved_evidence_specificity",
        lambda question, results: {
            "metric_match": True,
            "date_match": True,
            "reason": "Test specificity report.",
        },
    )
    monkeypatch.setattr(
        rag,
        "check_retrieved_evidence",
        lambda question, results: {
            "decision": decision,
            "reason": "Test gate decision.",
            "answerability": {
                "overlap_ratio": 1.0,
                "answerability_signal": True,
            },
            "specificity": {
                "specificity_signal": True,
            },
        },
    )

    def fake_gemini_client(api_key):
        client = FakeGeminiClient(api_key)
        fake_gemini_clients.append(client)
        return client

    monkeypatch.setattr(rag.genai, "Client", fake_gemini_client)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")

    return fake_gemini_clients


def test_allow_decision_calls_gemini(monkeypatch, capsys):
    clients = configure_common_mocks(monkeypatch, "ALLOW")
    monkeypatch.setattr("builtins.input", lambda prompt="": "What were net sales?")

    rag.main()

    assert len(clients) == 1
    assert clients[0].interactions.calls == 1
    assert "--- ANSWER ---" in capsys.readouterr().out


def test_review_decision_blocks_gemini(monkeypatch, capsys):
    clients = configure_common_mocks(monkeypatch, "REVIEW")
    monkeypatch.setattr("builtins.input", lambda prompt="": "What were net sales?")

    rag.main()

    assert clients == []
    output = capsys.readouterr().out
    assert "GENERATION STOPPED" in output
    assert "No answer was generated." in output


def test_decline_decision_blocks_gemini(monkeypatch, capsys):
    clients = configure_common_mocks(monkeypatch, "DECLINE")
    monkeypatch.setattr("builtins.input", lambda prompt="": "What were net sales?")

    rag.main()

    assert clients == []
    output = capsys.readouterr().out
    assert "GENERATION STOPPED" in output
    assert "No answer was generated." in output


def test_empty_retrieval_blocks_gemini(monkeypatch, capsys):
    clients = configure_common_mocks(monkeypatch, "ALLOW")
    monkeypatch.setattr(
        rag,
        "retrieve_chunks",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "What were net sales?")

    rag.main()

    assert clients == []
    output = capsys.readouterr().out
    assert "No relevant sources were retrieved." in output
    assert "No Gemini request was made." in output


def test_empty_question_blocks_gemini(monkeypatch, capsys):
    clients = configure_common_mocks(monkeypatch, "ALLOW")
    monkeypatch.setattr("builtins.input", lambda prompt="": "   ")

    rag.main()

    assert clients == []
    assert "Please enter a question." in capsys.readouterr().out
