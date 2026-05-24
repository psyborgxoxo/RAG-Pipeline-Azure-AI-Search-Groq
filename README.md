# RAG Pipeline — Azure AI Search + Groq

A production-grade Retrieval-Augmented Generation (RAG) pipeline built on Azure AI Search, Azure OpenAI Embeddings, and Groq LLaMA. Upload any document and ask questions about it — the system retrieves relevant context and generates grounded, cited answers.

---

## Architecture

```
Documents (PDF/Word/HTML/Image)
        │
        ▼
   ETL Parser          ← pdfplumber, python-docx, BeautifulSoup, pytesseract
        │
        ▼
 Semantic Chunker      ← 512 tokens, 50 overlap, metadata tagging
        │
        ▼
Azure OpenAI Embeddings ← text-embedding-ada-002 (1536-d vectors)
        │
        ▼
 Azure AI Search       ← Hybrid retrieval: Vector (HNSW) + BM25 keyword
        │
        ▼
   Groq LLaMA          ← llama-3.1-8b-instant, grounded generation
        │
        ▼
  REST API (FastAPI)   ← /ingest  /query  /health
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vector Store | Azure AI Search (Standard S1) |
| Embeddings | Azure OpenAI text-embedding-ada-002 |
| LLM | Groq llama-3.1-8b-instant |
| ETL | pdfplumber, python-docx, BeautifulSoup, pytesseract |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| API | FastAPI + Uvicorn |
| Evaluation | Ragas, R@5, MRR |
| Language | Python 3.11 |

---

## Project Structure

```
rag-pipeline/
├── src/
│   ├── index_setup.py     # Creates Azure AI Search index with vector + BM25 fields
│   ├── etl_parser.py      # Multi-format document parser (PDF, Word, HTML, images)
│   ├── chunker.py         # Semantic chunking with metadata tagging
│   ├── embedder.py        # Generates embeddings and uploads chunks to Azure
│   ├── query.py           # Hybrid search (vector + keyword)
│   ├── guardrails.py      # Grounding validation and hallucination detection
│   ├── evaluator.py       # Retrieval evaluation (R@5, MRR)
│   ├── rag.py             # End-to-end RAG loop
│   └── main.py            # FastAPI REST API
├── data/                  # Document storage
├── tests/                 # Test files
├── .env                   # Environment variables (not committed)
├── .gitignore
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.11+
- Azure subscription with:
  - Azure AI Search resource (Standard S1 or higher)
  - Azure OpenAI resource with `text-embedding-ada-002` deployed
- Groq API key (free at console.groq.com)

### Installation

```bash
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory:

```env
# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_ADMIN_KEY=your_admin_key

# Azure OpenAI (embeddings)
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_KEY=your_openai_key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Groq (chat generation)
GROQ_API_KEY=your_groq_key
```

---

## Usage

### 1. Create the Azure AI Search index

```bash
python src/index_setup.py
```

### 2. Ingest a document

```bash
python src/embedder.py data/your_document.pdf
```

### 3. Ask a question

```bash
python src/rag.py "What is the main finding of this paper?"
```

### 4. Run the API

```bash
uvicorn src.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## API Reference

### `GET /health`
Returns service health status.

**Response:**
```json
{
  "status": "ok",
  "service": "RAG Pipeline API"
}
```

---

### `POST /ingest`
Upload a document for ingestion. Supported formats: `.pdf`, `.docx`, `.html`, `.jpg`, `.png`

**Request:** `multipart/form-data` with `file` field

**Response:**
```json
{
  "filename": "document.pdf",
  "chunks_indexed": 122,
  "status": "success"
}
```

---

### `POST /query`
Ask a question about ingested documents.

**Request:**
```json
{
  "question": "What is the coffee automaton?"
}
```

**Response:**
```json
{
  "question": "What is the coffee automaton?",
  "answer": "The coffee automaton is a simple stochastic cellular automaton...",
  "sources": ["document.pdf"]
}
```

---

## Evaluation Results

Evaluated on the Coffee Automaton paper (Aaronson et al., 2014) with 3 test queries:

| Metric | Score |
|---|---|
| R@5 (Recall at 5) | 1.00 |
| MRR (Mean Reciprocal Rank) | 1.00 |
| Retrieval quality | GOOD |

---

## Key Design Decisions

**Hybrid retrieval over pure vector search**
Combining BM25 keyword matching with vector similarity using Azure AI Search's Reciprocal Rank Fusion (RRF) scoring consistently outperforms either approach alone — especially for queries with specific technical terms.

**Chunk size of 512 tokens with 50-token overlap**
Balances context preservation with retrieval precision. Smaller chunks improve retrieval relevance; overlap prevents context loss at chunk boundaries.

**Groq LPUs for generation**
Groq's Language Processing Units deliver significantly lower inference latency compared to GPU-based providers, making the query pipeline fast enough for interactive use.

**Grounding validation before response delivery**
Every LLM response is checked against source chunks before being returned. Ungrounded responses are blocked and replaced with an explicit "could not find" message — preventing hallucinations from reaching the user.

---

## Guardrails

The pipeline includes two layers of output safety:

- **Grounding check** — verifies the response contains content traceable to retrieved source chunks
- **Hallucination detection** — flags responses where less than 20% of words overlap with source material

---

## Roadmap

- [ ] Docker containerization
- [ ] Azure Container Apps deployment
- [ ] GPT-4o-mini integration (pending quota approval)
- [ ] Streaming responses via Server-Sent Events
- [ ] Multi-document filtering by source
- [ ] Conversation history support

---

## Author

**Sourav** — QA Automation Engineer / SDET  
Built as part of an AI Engineering portfolio project demonstrating enterprise RAG architecture on Azure.