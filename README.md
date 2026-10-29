# 🪔 Ramayana AI — Ask the Epic

> **A complete, end-to-end Retrieval-Augmented Generation (RAG) chatbot grounded in the complete Valmiki Ramayana (Gita Press, Gorakhpur — English translation).**  
> Answers are not invented — they are retrieved from the indexed scripture and generated with full Kanda/Canto source citations.

---

## What's New ✨

- **Full Valmiki Ramayana indexed** — All 7 Kandas (2,303 pages, Gita Press English edition) are embedded and searchable.
- **Kanda & Canto metadata** — Every retrieved chunk is tagged with its Kanda (book), Canto number, and page, giving rich citations like *"Sundarakanda (Book 5) · Canto XII · p. 1205"*.
- **Sanskrit page filtering** — Pages that are ≥ 60 % Devanagari are automatically skipped; only English translation text is indexed for clean retrieval.
- **Grounded system prompt** — The LLM is explicitly told it has the complete 7-Kanda Valmiki Ramayana and should cite by Kanda/Canto.

---

## Architecture

```mermaid
flowchart TD
    U([👤 User Question]) --> QP[Query Processing\n& Normalisation]
    QP --> QR{Follow-up\nDetected?}
    QR -- Yes --> RW[LLM Query\nRewriting]
    QR -- No --> EMB
    RW --> EMB[Embedding\nBAAI/bge-small-en-v1.5]
    EMB --> VDB[(ChromaDB\nPersistent Vector Store)]
    VDB --> RET[Top-K Chunk\nRetrieval]
    RET --> FLT[Similarity\nThreshold Filter]
    FLT --> CTX[Context\nConstruction]
    CTX --> GEN[Groq LLM\nGrounded Generation]
    GEN --> CIT[Citation\nValidation]
    CIT --> UI([🖥️ Streamlit\nAnswer + Sources])
    
    style U fill:#800020,color:#FFF8EE
    style UI fill:#800020,color:#FFF8EE
    style VDB fill:#1A3A5C,color:#FFF8EE
    style GEN fill:#5C3A1A,color:#FFF8EE
```

---

## How RAG Works Here

Unlike a plain chatbot that relies purely on the LLM's pretrained memory, this application:

1. **Indexes your documents** — The Valmiki Ramayana PDF is split into overlapping chunks, embedded using a local sentence-transformer model, and stored in a persistent ChromaDB vector database.
2. **Retrieves relevant passages** — Your question is embedded and compared against all indexed chunks. The most semantically similar passages are retrieved.
3. **Constructs grounded context** — Retrieved passages (not the whole book) are assembled with Kanda/Canto source labels and passed to the LLM.
4. **Generates a cited answer** — The LLM is explicitly instructed to use only the retrieved evidence, cite sources, and acknowledge when evidence is insufficient.
5. **Validates citations** — The system cross-checks every `[Source N]` citation against the actual retrieved chunks to prevent hallucinated references.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| UI | Streamlit |
| Embeddings | sentence-transformers (`BAAI/bge-small-en-v1.5`) |
| Vector Store | ChromaDB (persistent local) |
| LLM | Groq API (`qwen/qwen3.8-27b`) |
| Alt LLM | Ollama (local) |
| Document Loaders | pypdf (with Kanda/Canto metadata extraction) |
| Configuration | python-dotenv |
| Testing | pytest |

---

## Knowledge Base — The Seven Kandas

The chatbot is powered by the **complete Valmiki Ramayana** (Gita Press, Gorakhpur — English translation):

| # | Kanda | Content |
|---|-------|---------|
| 1 | **Balakanda** | Birth and childhood of Rama; Vishwamitra's hermitage; Sita Swayamvara |
| 2 | **Ayodhyakanda** | Life in Ayodhya; Kaikeyi's boons; 14-year exile of Rama |
| 3 | **Aranyakanda** | Forest exile; Surpanakha; abduction of Sita by Ravana |
| 4 | **Kishkindhakanda** | Alliance with Sugriva; killing of Vali; Vanara search parties |
| 5 | **Sundarakanda** | Hanuman's leap to Lanka; discovery of Sita; burning of Lanka |
| 6 | **Yuddhakanda** | Construction of Rama Setu; war with Ravana; Rama's victory |
| 7 | **Uttarakanda** | Rama's reign; Sita's exile; Lava and Kusha; epilogue |

---

## Project Structure

```
ramayana-rag-chatbot/
├── app.py                    ← Streamlit entry point
├── requirements.txt
├── .env.example              ← Copy to .env and fill in your key
├── .gitignore
├── pytest.ini
├── README.md
│
├── data/
│   ├── README.md             ← Source doc guidance & attribution
│   └── documents/
│       └── demo_ramayana.txt ← Public-domain fallback corpus (Griffith 1870)
│       [Place your Valmiki Ramayana PDF here — not committed to git]
│
├── vectorstore/
│   └── chroma/               ← Persistent ChromaDB (auto-created on ingestion)
│
├── src/
│   ├── config.py             ← Centralised settings from env vars
│   ├── document_loader.py    ← PDF/TXT/MD loaders with Kanda/Canto metadata
│   ├── text_splitter.py      ← Overlapping chunk splitter
│   ├── embeddings.py         ← Sentence-Transformer wrapper
│   ├── vector_store.py       ← ChromaDB CRUD
│   ├── retriever.py          ← Dense retrieval + deduplication
│   ├── query_rewriter.py     ← Follow-up question rewriting
│   ├── prompt_templates.py   ← Grounded system/user prompts (Valmiki-aware)
│   ├── generation.py         ← Groq + Ollama clients (streaming)
│   ├── rag_pipeline.py       ← End-to-end orchestration
│   ├── citations.py          ← Citation extraction & validation
│   ├── evaluation.py         ← Recall@k, Precision@k, MRR metrics
│   └── utils.py              ← Text cleaning, hashing, token counting, labels
│
├── scripts/
│   ├── ingest_documents.py   ← CLI ingestion pipeline
│   ├── rebuild_index.py      ← Full index rebuild
│   └── evaluate_rag.py       ← Evaluation suite
│
└── tests/
    ├── test_document_loader.py
    ├── test_text_splitter.py
    ├── test_retriever.py
    ├── test_citations.py
    └── test_rag_pipeline.py
```

---

## Prerequisites

- **Python 3.10 or higher** (3.11 recommended)
- **Windows 10/11** (PowerShell)
- **Groq API key** (free at [console.groq.com](https://console.groq.com)) — required for answer generation only
- **Internet connection** — for the first embedding model download (~130 MB)
- **Valmiki Ramayana PDF** — place in `data/documents/` (not included in repo due to file size; see note below)

---

## Installation (Windows PowerShell)

### 1. Clone the repository
```powershell
git clone https://github.com/KOTHAPU/RAG-Based-Ramayana-Chatbot.git
cd RAG-Based-Ramayana-Chatbot
```

### 2. Create a virtual environment
```powershell
python -m venv venv
```

### 3. Activate the virtual environment
```powershell
.\venv\Scripts\Activate.ps1
```
> If you encounter an execution policy error on Windows, run:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 4. Install dependencies
```powershell
pip install -r requirements.txt
```

### 5. Configure environment variables
```powershell
Copy-Item .env.example .env
```

Add your Groq API key in `.env`:
```
GROQ_API_KEY=gsk_your_key_here
```

---

## Adding the Valmiki Ramayana PDF

The PDF is **not included in the repository** (excluded via `.gitignore` — too large for git). To use the full knowledge base:

1. Place your copy of the Valmiki Ramayana PDF in `data/documents/`.
2. The recommended edition is the **Gita Press, Gorakhpur** English translation.
3. Run the ingestion script:

```powershell
python scripts/ingest_documents.py --rebuild
```

> The `.gitignore` rule `data/documents/*.pdf` ensures the book is never accidentally committed to the repository.

---

## Ingesting Documents

### First-time / full rebuild (recommended after adding the PDF)
```powershell
python scripts/ingest_documents.py --rebuild
```

### Incremental ingestion (skips unchanged files)
```powershell
python scripts/ingest_documents.py
```

### Check index status
```powershell
python -c "from src.config import settings; from src.vector_store import VectorStore; vs = VectorStore(settings.vectorstore_dir, settings.chroma_collection_name); print(f'Indexed chunks: {vs.count()}')"
```

> **What happens during ingestion:**
> - Kanda (book) and Canto are detected automatically from PDF page headers
> - Devanagari-only (Sanskrit) pages are skipped; only the English translation is indexed
> - Pages and chunks are deduplicated by content hash

---

## Launching the Chatbot

```powershell
streamlit run app.py
```

The application opens in your browser at `http://localhost:8501`.

---

## Configuring Groq

1. Create a free API key at [console.groq.com](https://console.groq.com).
2. Add it to your `.env` file (`GROQ_API_KEY=gsk_...`).
3. The recommended default model is `qwen/qwen3.8-27b`.

---

## Configuring Ollama (Local Alternative)

1. Install Ollama from [ollama.ai](https://ollama.ai).
2. Pull a model: `ollama pull llama3`
3. In `.env`, set:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3
   ```

---

## Running Tests

```powershell
pytest tests/ -v
```

Tests that do **not** require external services or a populated index:
- `test_document_loader.py` — file loading, metadata, deduplication
- `test_text_splitter.py` — chunking, overlap, token counts
- `test_citations.py` — citation extraction and validation
- `test_retriever.py` — retriever logic (mocked vector store)
- `test_rag_pipeline.py` — pipeline orchestration (mocked dependencies)

---

## Evaluating Retrieval Quality

Requires a populated index:
```powershell
python scripts/evaluate_rag.py --top-k 5
```

This reports:
- **Recall@5** — fraction of expected keywords retrieved
- **Precision@5** — fraction of retrieved content that was relevant
- **Hit Rate** — fraction of queries with at least one relevant result
- **MRR** — Mean Reciprocal Rank

---

## Common Errors & Troubleshooting

| Error | Solution |
|-------|----------|
| `GROQ_API_KEY not set` | Add key to `.env` file |
| `Vector store is empty` | Run `python scripts/ingest_documents.py` |
| `Failed to load embedding model` | Check internet connection; model downloads on first use |
| `chromadb` version conflict | Run `pip install chromadb --upgrade` |
| `Execution policy` error | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| App crashes on startup | Activate venv first: `.\venv\Scripts\Activate.ps1` |
| PDF not found during ingestion | Place the Valmiki Ramayana PDF in `data/documents/` |

---

## Privacy & Data Handling

- All document embeddings are stored locally in `vectorstore/chroma/`.
- No document text is sent externally except as part of the LLM context window sent to Groq.
- API keys are read from environment variables and never exposed in the UI.
- The Valmiki Ramayana PDF is excluded from git (`.gitignore`) and is never uploaded to the repository.

---

## Source Attribution

The knowledge base uses:
- **Valmiki Ramayana** — English translation published by **Gita Press, Gorakhpur, India**.  
  Covers all 7 Kandas of the original Sanskrit epic composed by the sage Valmiki.

The demo fallback dataset (`demo_ramayana.txt`) is drawn from:
- Griffith, Ralph T.H. (translator). *The Ramayana of Valmiki*. Benares: E.J. Lazarus and Co., 1870–1874. (Public domain)

---

## Known Limitations

- Answer quality depends on the quality and coverage of indexed documents.
- Follow-up query rewriting uses an LLM call — when the API is unavailable, heuristic rewriting is used.
- Generation requires a Groq API key or a running Ollama instance.
- Sanskrit verse numbers and diacritical marks in the Gita Press PDF may not always be extracted cleanly by pypdf.

---

## Future Improvements

- Hybrid BM25 + dense retrieval for better keyword precision
- Cross-encoder reranking for higher retrieval quality
- Multi-document QA with explicit tradition/edition comparison
- Persistent conversation history across sessions
- Support for Sanskrit Devanagari script documents
