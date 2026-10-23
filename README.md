# Enterprise AI Document Intelligence

An evidence-grounded question-answering application that uses Retrieval-Augmented Generation (RAG) to answer questions about enterprise documents with source-page citations and evidence validation.

## Overview

Enterprise AI Document Intelligence enables users to ask natural-language questions about an indexed company report and receive answers grounded in retrieved document evidence.

The project uses the Volvo Group Annual Report 2025 as its example document. It combines semantic search, large language model generation, citation handling, and answer validation in a FastAPI application with a simple browser-based interface.

## Features

- **Document question answering:** Ask natural-language questions about the indexed annual report.
- **Retrieval-Augmented Generation (RAG):** Retrieve relevant document passages before generating an answer.
- **Semantic search:** Use embeddings to find relevant content in Qdrant.
- **Source-page citations:** Display page references associated with supporting evidence.
- **Evidence validation:** Check citations and numerical claims against retrieved evidence.
- **REST API:** Expose question-answering and operational endpoints through FastAPI.
- **Web interface:** Submit questions and view answers and source pages in a browser.
- **Operational visibility:** Provide health, readiness, and metrics endpoints.

## Architecture

```text
User
  |
  v
Web Interface (HTML, CSS, JavaScript)
  |
  v
FastAPI Application
  |
  v
Question Processing
  |
  v
Embedding Model
  |
  v
Qdrant Vector Search
  |
  v
Retrieved Document Evidence
  |
  v
Gemini Answer Generation
  |
  v
Citation and Evidence Validation
  |
  v
Answer + Source Pages + Validation Result
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| API framework | FastAPI |
| Vector database | Qdrant |
| Embeddings | Sentence Transformers |
| Language model | Google Gemini |
| Data validation | Pydantic |
| Frontend | HTML, CSS, JavaScript |
| Testing | pytest |
| API documentation | OpenAPI / Swagger UI |

## Project Structure

```text
enterprise-ai-document-intelligence/
├── data/
│   └── <annual-report-pdf>
├── src/
│   ├── main.py
│   ├── api_service.py
│   ├── rag.py
│   ├── search.py
│   ├── embed_and_store.py
│   ├── ingest.py
│   ├── chunker.py
│   ├── answer_validation.py
│   ├── answerability.py
│   ├── evidence_gate.py
│   ├── evidence_specificity.py
│   ├── rate_limiter.py
│   └── metrics.py
├── static/
│   └── index.html
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

The structure above highlights the main application components. Some files may differ depending on the current repository version.

## Prerequisites

- Python 3.10 or a compatible version
- A Google Gemini API key
- A running Qdrant instance
- Git

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/BushraTaz/enterprise-ai-document-intelligence.git
cd enterprise-ai-document-intelligence
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

If the project dependencies have changed, ensure `requirements.txt` reflects the packages required by the current codebase.

### 4. Configure environment variables

Create a `.env` file in the project root and add your Gemini API key using the environment variable name expected by the application.

Example:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

Replace the placeholder with your own key. Never commit `.env` or expose API keys in source code, screenshots, logs, or public documentation.

### 5. Start Qdrant

Start a Qdrant instance and ensure it is reachable at the URL configured by the application.

For a local setup, the default URL may be:

```text
http://localhost:6333
```

Ensure the expected vector collection has been populated before asking questions.

### 6. Prepare the document index

Use the project's ingestion and embedding scripts to process the annual report and store its document chunks and embeddings in Qdrant.

Check the configuration and existing collection before running ingestion. Avoid recreating or deleting a populated collection unless you intentionally want to rebuild the index.

## Running the Application

From the repository root, start the FastAPI server:

```powershell
python -m uvicorn main:app --app-dir src
```

Open the web interface:

**http://127.0.0.1:8000/**

Open the interactive API documentation:

**http://127.0.0.1:8000/docs**

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serve the web interface |
| `/api/v1/ask` | POST | Ask a question about indexed documents |
| `/health` | GET | Check application health |
| `/ready` | GET | Check application readiness |
| `/metrics` | GET | Expose application metrics |

Operational endpoints may be hidden from the interactive API documentation.

### Example question request

```http
POST /api/v1/ask
Content-Type: application/json
```

```json
{
  "question": "What were Volvo Group's total net sales in 2025 and 2024?"
}
```

A successful response contains a generated answer, source-page references, and a validation status. The exact response structure depends on the current API implementation.

Example response shape:

```json
{
  "status": "answered",
  "answer": "Volvo Group's total net sales were SEK 479,183 M in 2025 and SEK 526,816 M in 2024 [Page 37, Page 87].",
  "citations": [
    {
      "page": 37
    },
    {
      "page": 87
    }
  ],
  "validation_passed": true
}
```

The figures and page references in this example come from a test of the Volvo Group Annual Report 2025. Model-generated answers should always be interpreted in light of their retrieved evidence and validation result.

## Evidence Validation

The application includes validation logic intended to improve answer reliability by checking:

- Whether citations are present.
- Whether cited pages correspond to retrieved evidence.
- Whether numerical claims are supported by the available evidence.
- Whether validation checks pass consistently.

A successful validation result indicates that the implemented checks passed. It does not guarantee that every generated statement is correct, so important claims should still be verified against the original document.

## Testing

Run the test suite from the repository root:

```powershell
python -m pytest -v
```

Additional retrieval, evaluation, and diagnostic scripts are available in the source directory. Review each script before running it, especially if it makes external API calls or modifies stored data.

## Current Limitations

- Response time can vary depending on retrieval, model generation, and network latency.
- Answer quality depends on the indexed document, retrieved evidence, and model output.
- Citation and numerical validation cannot guarantee complete factual correctness.
- The current example focuses on a single indexed annual report rather than a complete multi-document enterprise deployment.

## Potential Future Improvements

- Reduce end-to-end response latency.
- Add a clearer validation-status indicator to the web interface.
- Improve retrieval evaluation and relevance ranking.
- Add document upload and indexing workflows.
- Support multiple documents and metadata-based filtering.
- Add authentication, authorization, and production deployment configuration.
- Expand automated tests for API behavior, citations, and unsupported questions.

## Security Notes

- Store API keys in environment variables or a secrets manager.
- Keep `.env` and other sensitive configuration files out of Git.
- Avoid logging secrets or sensitive document content.
- Review authentication, rate limiting, and access controls before deploying the application publicly.

## License

Add a license before distributing or reusing this project. Until a license is specified, do not assume that others have permission to reuse the repository's code.

---

**Built as an AI engineering portfolio project exploring evidence-grounded document question answering, vector search, and LLM application development.**
