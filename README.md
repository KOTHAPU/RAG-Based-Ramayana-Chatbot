# Ramayana RAG AI — Valmiki Ramayana QA System

A domain-specific Question-Answering assistant built on top of the original **Valmiki Ramayana** using Retrieval-Augmented Generation (RAG).

Standard LLMs often struggle with ancient texts—they tend to mix up regional retellings, TV adaptations, or simply make up chapter citations. This project addresses that problem by grounding answers in the complete English translation of the Valmiki Ramayana (Gita Press edition), providing source references (Kanda, Canto, and Page number) for factual claims.

---

## What is this project?

This system implements an end-to-end RAG architecture:
1. **Document Processing:** Ingests the 2,303-page Valmiki Ramayana PDF.
2. **Text Chunking & Embedding:** Breaks the text into overlapping chunks and embeds them using `BAAI/bge-small-en-v1.5`.
3. **Dense Retrieval:** Uses ChromaDB to look up passages relevant to user questions.
4. **Grounded Generation:** Prompts Groq's `qwen/qwen3.8-27b` with the retrieved passages, instructing it to answer strictly using the provided context and cite its sources.

---

## Dataset

- **Source Document:** *Valmiki Ramayana* (English translation)
- **Publication:** Gita Press, Gorakhpur
- **Scope:** All 7 Kandas (Books):
  - **Balakanda** (Childhood of Rama, Vishwamitra, Sita Swayamvara)
  - **Ayodhyakanda** (Events in Ayodhya, Kaikeyi's boons, exile)
  - **Aranyakanda** (Forest dwelling, Surpanakha, Sita's abduction)
  - **Kishkindhakanda** (Meeting Hanuman, alliance with Sugriva)
  - **Sundarakanda** (Hanuman's journey to Lanka, finding Sita)
  - **Yuddhakanda** (The war, construction of Rama Setu, defeat of Ravana)
  - **Uttarakanda** (Return to Ayodhya, subsequent events)

---

## Technical Pipeline

### 1. Document Extraction & Preprocessing
- Used `pypdf` to extract text from 2,303 pages.
- Filtered out pages with high Devanagari density (≥ 60%) to focus on the English translation.
- Extracted Kanda and Canto names from header patterns to preserve contextual metadata.

### 2. Chunking & Indexing
- **Chunk Size:** 800 tokens
- **Chunk Overlap:** 120 tokens (preserves continuity across chunk borders)
- **Embedding Model:** `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors)
- **Vector Database:** ChromaDB (persistent local storage)
- **Total Indexed Chunks:** 7,420 chunks across 232 embedding batches

### 3. Generation & Guardrails
- **Inference:** Groq API (`qwen/qwen3.8-27b`)
- **Temperature:** `0.1` (low temperature to minimize creative variance)
- **Fallback Rule:** If the retrieved passages lack relevant details, the model responds with an explicit insufficient evidence notice.
- **Citation Validator:** Validates extracted `[Source N]` tags against retrieved chunk metadata.

---

## System Metrics & Observations

### Data Ingestion Summary
- **Total Pages in Source:** 2,303
- **Processed Sections:** 2,300 (blank leaves skipped)
- **Total Chunks in Index:** 7,420
- **Ingestion Duration:** ~14.4 minutes (863 seconds)
- **Database Size:** ~91.5 MB (`chroma.sqlite3`)

### Evaluation Concepts Tracked
- **Hit Rate:** Proportion of queries where at least one relevant passage is retrieved among top results (target: > 90%).
- **Mean Reciprocal Rank (MRR):** Measures the ranking position of the first relevant chunk (target: > 0.75).
- **Recall@5:** Ratio of expected key details present in the top-5 chunks relative to total expected facts.
- **Precision@5:** Ratio of retrieved chunks that contain useful information.
- **Citation Validity:** Percentage of generated `[Source N]` citations that point to retrieved chunks.

---

## Tech Stack

- **Python:** 3.11
- **UI:** Streamlit
- **Vector Database:** ChromaDB
- **Embeddings:** `sentence-transformers` (`BAAI/bge-small-en-v1.5`)
- **LLM API:** Groq (`qwen/qwen3.8-27b`)
- **PDF Extraction:** PyPDF
- **Testing:** Pytest

---

## Setup & Running Locally

### 1. Clone the repository
```powershell
git clone https://github.com/KOTHAPU/RAG-Based-Ramayana-Chatbot.git
cd RAG-Based-Ramayana-Chatbot
```

### 2. Create and activate a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file from the example:
```powershell
Copy-Item .env.example .env
```
Add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Launch the Application
The vector store is pre-indexed in the repository under `vectorstore/chroma/`, so the app can be run directly:
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Tests & Evaluation

To run test suites:
```powershell
pytest tests/ -v
```

To run the retrieval evaluation script:
```powershell
python scripts/evaluate_rag.py --top-k 5
```
