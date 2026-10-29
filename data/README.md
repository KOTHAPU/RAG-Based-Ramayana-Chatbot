# Source Documents for the Ramayana RAG Chatbot

This directory holds the text corpus for the chatbot. Add your Ramayana source documents here, then run the ingestion script.

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Page numbers extracted automatically |
| Plain text | `.txt` | UTF-8 preferred |
| Markdown | `.md` | Headings used as section metadata |

## Recommended Public-Domain Sources

The following are legally accessible, public-domain Ramayana texts whose reuse is permitted:

### 1. Sacred Texts Archive (sacred-texts.com)
- **Valmiki Ramayana** translated by Ralph T. H. Griffith (1870–1874)  
  URL: https://sacred-texts.com/hin/rama/index.htm  
  Format: HTML (save individual Kandas as `.txt` files)  
  License: Public domain (pre-1928 translation)

- **Valmiki Ramayana** translated by M. N. Dutt (1891–1894)  
  URL: https://sacred-texts.com/hin/dutt/index.htm  
  License: Public domain

### 2. Project Gutenberg (gutenberg.org)
Search for "Ramayana" at https://www.gutenberg.org/ebooks/search/?query=ramayana  
Download `.txt` format for easiest ingestion.

Notable titles:
- *The Ramayana and Mahabharata* condensed by Romesh Chunder Dutt (1900)  
  https://www.gutenberg.org/ebooks/16295

### 3. Wikisource
- Valmiki Ramayana (various translations): https://en.wikisource.org/wiki/The_Ramayana  
  Copy individual Kanda pages as Markdown or plain text.

## Attribution Metadata

When you place a document in this folder, create a companion `.txt` file named `<filename>_attribution.txt` containing:

```
Title: Valmiki Ramayana – Bala Kanda
Translator: Ralph T. H. Griffith
Year: 1870
Source: sacred-texts.com
License: Public domain
Notes: Original Sanskrit by Valmiki
```

The ingestion pipeline will embed this attribution as document metadata where available.

## Edition Notes

- The Ramayana exists in many regional traditions (Valmiki, Kamba, Tulsi Das, etc.) and translations.
- Different editions may number verses, chapters, and kandas differently.
- The chatbot will note disagreements between sources when they are retrieved together.
- Do **not** add copyrighted or restricted materials without verifying their license.

## Demo Dataset

A small demo dataset (`demo_ramayana.txt`) is included in this directory. It contains carefully curated, public-domain excerpts from the Griffith translation (1870–1874) clearly attributed to that source. These excerpts allow you to test the full RAG pipeline before sourcing larger documents.

**Important**: The demo dataset is a small sample — only a fraction of the full epic. For comprehensive coverage, add complete Kanda texts as described above.

## Ingestion Commands

```powershell
# Activate virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# Ingest all documents (incremental — skips unchanged files)
python scripts/ingest_documents.py

# Full rebuild (drops and re-indexes everything)
python scripts/ingest_documents.py --rebuild

# Check how many chunks are indexed
python -c "from src.config import settings; from src.vector_store import VectorStore; vs = VectorStore(settings.vectorstore_dir, settings.chroma_collection_name); print(f'Indexed chunks: {vs.count()}')"
```
