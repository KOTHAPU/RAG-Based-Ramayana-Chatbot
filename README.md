# 🪔 Ramayana RAG AI — Ask the Epic

Hey everyone! 👋 Welcome to my college project on building a domain-specific **Retrieval-Augmented Generation (RAG)** chatbot powered by the sacred ancient epic — the **Valmiki Ramayana**.

Instead of relying on a normal generic AI chatbot that often hallucinates, mixes up TV show stories with regional folklores, or invents fake chapter numbers, I built an end-to-end question-answering system. It strictly searches through an authentic digital translation of the Valmiki Ramayana and gives answers backed by **exact book citations (Kanda, Canto, and Page number)**.

---

## 📌 Project Overview & Motivation

### Why did I build this?
General LLMs (like standard ChatGPT or Llama) have heard of the Ramayana, but:
1. They mix up different versions (Tulsidas Ramcharitmanas, regional folk stories, TV serial interpretations, etc.) with the original Valmiki text.
2. They cannot tell you which specific chapter or verse a statement comes from.
3. They often guess or fabricate details when they are not sure.

### The Solution:
By implementing **Retrieval-Augmented Generation (RAG)**:
- My system breaks the book into searchable chunks and converts them into mathematical embeddings.
- When you ask a question, it finds the most relevant passages from the book first.
- It then hands those exact passages to an LLM (via Groq API) with strict instructions: **"Answer ONLY using this text. If it is not in the text, admit you don't know."**

---

## 📚 Dataset Details

I completely replaced initial sample demo snippets with the complete authoritative English edition:

- **Source:** *Valmiki Ramayana* (English translation)
- **Publisher:** Gita Press, Gorakhpur, India
- **File Format:** PDF (`valmiki-ramayana-gita-press-english.pdf`)
- **Total Pages:** 2,303 pages
- **Coverage:** All **7 Kandas (Books)** of the epic:
  1. **Balakanda** (Childhood of Lord Rama, Vishwamitra, Sita Swayamvara)
  2. **Ayodhyakanda** (Palace events in Ayodhya, Kaikeyi's boons, exile)
  3. **Aranyakanda** (Forest life, Surpanakha, golden deer, Sita's abduction)
  4. **Kishkindhakanda** (Meeting Hanuman, alliance with Sugriva, search for Sita)
  5. **Sundarakanda** (Hanuman's leap to Lanka, finding Sita in Ashoka Vatika)
  6. **Yuddhakanda** (Construction of Rama Setu bridge, the war, killing of Ravana)
  7. **Uttarakanda** (Return to Ayodhya, coronation, later life)

---

## ⚙️ What I Built & How It Works

### 1. Document Extraction & Cleaning
- Extracted text from all 2,303 pages using `pypdf`.
- **Sanskrit Filter:** Built regex checks to skip pages that are mostly Sanskrit/Devanagari scripts (≥ 60%) so the vector store contains clean, search-friendly English translation text.
- **Metadata Tagging:** Wrote header-parser rules to detect which Kanda and Canto each page belongs to and attached this metadata to every chunk.

### 2. Chunking & Overlap
- **Chunk Size:** 800 tokens
- **Chunk Overlap:** 120 tokens
- *Why overlap?* An overlap prevents splitting sentences or critical story contexts right in the middle of an action.

### 3. Embeddings & Vector Database
- **Embedding Model:** `BAAI/bge-small-en-v1.5` (via `sentence-transformers`, 384-dimensional dense vectors).
- **Vector Database:** `ChromaDB` (persistent local storage).
- Total processed chunks: **7,420 chunks**.

### 4. Grounded Generation & Guardrails
- **LLM Provider:** Groq API (`qwen/qwen3.8-27b`) with fallback to local Ollama.
- **Low Temperature:** `0.1` (keeps responses strictly factual).
- **Citation Validator:** A custom regex parser in `src/citations.py` verifies that every `[Source N]` mentioned by the model actually matches an existing retrieved passage.

---

## 📊 Evaluation & System Metrics

Here are the real metrics recorded from building and testing this project:

### 1. Ingestion & Embedding Stats
| Metric | Value |
| :--- | :--- |
| **Total PDF Pages** | 2,303 pages |
| **Document Sections Loaded** | 2,300 sections (excluding blanks) |
| **Total Vector Chunks** | 7,420 chunks |
| **Batches Processed** | 232 batches |
| **Embedding Time** | ~14.4 minutes (863.4 seconds) |
| **Database File Size** | ~91.5 MB (`chroma.sqlite3`) |

### 2. What the Evaluation Metrics Mean (And What We Aim For)

To measure how well the RAG pipeline performs, I evaluated it across both **search accuracy** and **hallucination prevention**:

#### A. Retrieval Metrics (Search Quality)
- **Hit Rate (Target: > 90%):** Measures whether at least one relevant passage from the book was found in the top-5 results.
- **Mean Reciprocal Rank / MRR (Target: > 0.75):** Measures how high up the best answer was. If the most relevant chunk is ranked #1, reciprocal rank is $1.0$.
- **Recall@5 (Target: > 80%):** The percentage of important facts needed for the question that were successfully brought into the prompt.
- **Precision@5 (Target: > 60%):** The fraction of the 5 retrieved chunks that directly contained useful signal rather than surrounding background text.

#### B. Generation & Hallucination Metrics (Answer Quality)
- **Hallucination Rate (Target: < 2%):** The percentage of statements made by the LLM that cannot be proven from the retrieved passage. If the text does not contain the answer, the model is prompted to state: `[INSUFFICIENT_EVIDENCE]`.
- **Citation Validity Rate (Achieved: 100%):** Every citation generated by the model must match a real passage retrieved from ChromaDB.

---

## 💻 Tech Stack Used

- **Programming Language:** Python 3.11
- **User Interface:** Streamlit (custom Ramayana-themed UI)
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace `sentence-transformers` (`BAAI/bge-small-en-v1.5`)
- **LLM Inference:** Groq Cloud (`qwen/qwen3.8-27b`)
- **Document Processing:** PyPDF & Python Regular Expressions
- **Testing:** Pytest

---

## 🚀 How to Run the Project Locally

### 1. Clone the repository
```powershell
git clone https://github.com/KOTHAPU/RAG-Based-Ramayana-Chatbot.git
cd RAG-Based-Ramayana-Chatbot
```

### 2. Set up Python Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install requirements
```powershell
pip install -r requirements.txt
```

### 4. Setup your API Key
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` and paste your free Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 5. Launch the Streamlit Web App
Because the 7,420 chunks are already embedded and committed to the `vectorstore/` folder, you don't have to wait 15 minutes to re-embed! You can start the app immediately:
```powershell
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧪 Running Tests & Evaluation

To run the unit tests:
```powershell
pytest tests/ -v
```

To run the retrieval evaluation script:
```powershell
python scripts/evaluate_rag.py --top-k 5
```

---

## 🎓 What I Learned from this Project

1. **How RAG eliminates hallucinations:** Fine-tuning teaches style, but RAG provides ground-truth facts. Providing context directly in the prompt is the best way to prevent made-up information.
2. **Chunking matters a lot:** Too small chunks lose the narrative flow; too big chunks bring noise. An 800-token window with 120-token overlap was the sweet spot for epic verses.
3. **Data cleaning is 80% of the work:** Parsing a 2,300-page bilingual PDF required stripping diacritics, skipping devanagari-only pages, and extracting Kanda headers.
4. **Deploying Vector Databases:** Learning how to vacuum and optimize ChromaDB SQLite files to fit under GitHub repository limits (~91.5 MB) so the app works seamlessly on Streamlit Cloud.

---
*Created as an applied AI / NLP student project.* 🪔
