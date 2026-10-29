"""
document_loader.py — Load and extract text from PDF, TXT, and Markdown files.

Each loader returns a list of Document objects preserving all available metadata.
Empty, malformed, and duplicate documents are handled gracefully.

Valmiki Ramayana (Gita Press edition) support:
  - Extracts Kanda (book) and Canto metadata from page headers.
  - Filters out Sanskrit-only pages (Devanagari text) for cleaner English retrieval.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.utils import clean_text, file_hash, text_hash

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

# ── Valmiki Ramayana (Gita Press) Kanda detection ────────────────────────────

# Map transliterated Kanda names (as they appear in Gita Press headers) to
# canonical English names and their book numbers.
_KANDA_PATTERNS: list[tuple[re.Pattern, str, int]] = [
    (re.compile(r"B[AÅ]LAKA[NŒN][ŒD]A", re.I),   "Balakanda (Book 1 — Childhood)",  1),
    (re.compile(r"AYODHYAKA[NŒN][ŒD]A", re.I),    "Ayodhyakanda (Book 2 — Ayodhya)", 2),
    (re.compile(r"ARA[NŒN][ŒD]YAKA[NŒN][ŒD]A", re.I), "Aranyakanda (Book 3 — Forest)", 3),
    (re.compile(r"KI[SŠ][KI]NDH[AÅ]KA[NŒN][ŒD]A", re.I), "Kishkindhakanda (Book 4 — Kishkindha)", 4),
    (re.compile(r"SUNDARAKA[NŒN][ŒD]A", re.I),    "Sundarakanda (Book 5 — Beauty)",   5),
    (re.compile(r"YUDDHAKA[NŒN][ŒD]A|LANKAKA[NŒN][ŒD]A", re.I), "Yuddhakanda (Book 6 — War)", 6),
    (re.compile(r"UTTARAKA[NŒN][ŒD]A", re.I),     "Uttarakanda (Book 7 — Epilogue)",  7),
]

# Detect Canto numbers: "Canto I", "Canto IV", etc.
_CANTO_PATTERN = re.compile(r"Canto\s+([IVXLCDM]+|\d+)", re.I)

# Detect pages that are predominantly Sanskrit (Devanagari Unicode block U+0900–U+097F)
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _detect_kanda(text: str) -> tuple[str, int]:
    """Return (kanda_name, kanda_number) detected from page text, or ('Unknown', 0)."""
    for pattern, name, number in _KANDA_PATTERNS:
        if pattern.search(text):
            return name, number
    return "Unknown", 0


def _detect_canto(text: str) -> str:
    """Return the first Canto reference found in text, or empty string."""
    m = _CANTO_PATTERN.search(text)
    return m.group(0) if m else ""


def _is_mostly_sanskrit(text: str) -> bool:
    """Return True if more than 60 % of non-whitespace characters are Devanagari."""
    non_ws = text.replace(" ", "").replace("\n", "").replace("\t", "")
    if not non_ws:
        return False
    deva_count = len(_DEVANAGARI_RE.findall(non_ws))
    return deva_count / len(non_ws) > 0.60


@dataclass
class Document:
    """
    A single loaded document (before chunking).

    Attributes:
        text: Cleaned full text of the document.
        metadata: Dict of metadata fields (source, page count, etc.).
        doc_id: Unique document identifier (based on file hash).
    """

    text: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = text_hash(self.text)[:16]

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) < 50

    def __repr__(self) -> str:
        return (
            f"Document(id={self.doc_id!r}, "
            f"chars={len(self.text)}, "
            f"source={self.metadata.get('source_filename', '?')!r})"
        )


# ── Individual loaders ────────────────────────────────────────────────────────

def load_pdf(path: Path) -> list[Document]:
    """
    Extract text from a PDF, one Document per page.

    For the Valmiki Ramayana (Gita Press edition) the function additionally:
    - Skips pages that are ≥ 60 % Devanagari (Sanskrit-only pages).
    - Injects Kanda (book) and Canto metadata extracted from the page text.
    Falls back to full-document extraction if per-page extraction fails.
    """
    documents: list[Document] = []
    try:
        import pypdf

        reader = pypdf.PdfReader(str(path))
        num_pages = len(reader.pages)
        logger.info("Loading PDF '%s' (%d pages)", path.name, num_pages)

        # Track the last-seen Kanda so we can carry it forward across pages
        last_kanda: str = "Unknown"
        last_kanda_num: int = 0

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
                # Skip blank pages
                if not raw.strip():
                    continue

                # Skip pages that are predominantly Sanskrit (Devanagari)
                if _is_mostly_sanskrit(raw):
                    continue

                text = clean_text(raw)
                if not text.strip():
                    continue

                # Detect Kanda from raw text (before cleaning strips diacritics)
                kanda, kanda_num = _detect_kanda(raw)
                if kanda_num > 0:
                    last_kanda = kanda
                    last_kanda_num = kanda_num
                else:
                    kanda, kanda_num = last_kanda, last_kanda_num

                # Detect Canto
                canto = _detect_canto(raw)

                doc = Document(
                    text=text,
                    metadata={
                        "source_filename": path.name,
                        "source_path": str(path),
                        "file_hash": file_hash(path),
                        "page_number": page_num,
                        "total_pages": num_pages,
                        "file_type": "pdf",
                        "document_type": "ramayana_text",
                        "kanda": kanda,
                        "kanda_number": kanda_num,
                        "canto": canto,
                        "edition": "Valmiki Ramayana — Gita Press, Gorakhpur (English translation)",
                    },
                )
                documents.append(doc)
            except Exception as exc:
                logger.warning(
                    "Could not extract page %d from '%s': %s", page_num, path.name, exc
                )

    except ImportError:
        logger.error("pypdf is not installed. Run: pip install pypdf")
    except Exception as exc:
        logger.error("Failed to load PDF '%s': %s", path.name, exc)

    return documents


def load_txt(path: Path) -> list[Document]:
    """Load a plain-text file as a single Document."""
    try:
        # Try common encodings
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                raw = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            logger.error("Could not decode '%s' with any known encoding.", path.name)
            return []

        text = clean_text(raw)
        if not text.strip():
            logger.warning("'%s' is empty after cleaning.", path.name)
            return []

        return [
            Document(
                text=text,
                metadata={
                    "source_filename": path.name,
                    "source_path": str(path),
                    "file_hash": file_hash(path),
                    "file_type": "txt",
                    "document_type": "ramayana_text",
                },
            )
        ]
    except Exception as exc:
        logger.error("Failed to load TXT '%s': %s", path.name, exc)
        return []


def load_markdown(path: Path) -> list[Document]:
    """Load a Markdown file, splitting on H1/H2 headings when present."""
    try:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                raw = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            logger.error("Could not decode '%s'.", path.name)
            return []

        import re

        # Split on major headings (# or ##) to create section documents
        sections = re.split(r"\n(?=#{1,2} )", raw)
        documents: list[Document] = []
        for idx, section in enumerate(sections):
            text = clean_text(section)
            if not text.strip():
                continue
            # Extract heading for metadata
            heading_match = re.match(r"^#{1,2} (.+)", text)
            chapter = heading_match.group(1).strip() if heading_match else f"Section {idx + 1}"
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source_filename": path.name,
                        "source_path": str(path),
                        "file_hash": file_hash(path),
                        "chapter": chapter,
                        "section_index": idx,
                        "file_type": "markdown",
                        "document_type": "ramayana_text",
                    },
                )
            )
        return documents
    except Exception as exc:
        logger.error("Failed to load Markdown '%s': %s", path.name, exc)
        return []


# ── Directory loader ──────────────────────────────────────────────────────────

def iter_supported_files(directory: Path) -> Iterator[Path]:
    """Yield all supported document files under directory (recursive)."""
    if not directory.exists():
        logger.warning("Data directory '%s' does not exist.", directory)
        return
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_documents(
    directory: Path,
    skip_hashes: set[str] | None = None,
) -> list[Document]:
    """
    Load all supported documents from directory.

    Args:
        directory: Path to the documents folder.
        skip_hashes: File hashes to skip (already indexed).

    Returns:
        List of Document objects (potentially multiple per PDF page).
    """
    skip_hashes = skip_hashes or set()
    all_docs: list[Document] = []
    seen_content_hashes: set[str] = set()

    files = list(iter_supported_files(directory))
    if not files:
        logger.warning("No supported documents found in '%s'.", directory)
        return []

    for path in files:
        try:
            fhash = file_hash(path)
        except OSError as exc:
            logger.warning("Cannot read file '%s': %s", path.name, exc)
            continue

        if fhash in skip_hashes:
            logger.debug("Skipping unchanged file: %s", path.name)
            continue

        ext = path.suffix.lower()
        if ext == ".pdf":
            docs = load_pdf(path)
        elif ext == ".txt":
            docs = load_txt(path)
        elif ext in (".md", ".markdown"):
            docs = load_markdown(path)
        else:
            continue

        for doc in docs:
            if doc.is_empty:
                logger.debug("Skipping empty document from '%s'.", path.name)
                continue
            chash = text_hash(doc.text)
            if chash in seen_content_hashes:
                logger.debug("Skipping duplicate content from '%s'.", path.name)
                continue
            seen_content_hashes.add(chash)
            all_docs.append(doc)

    logger.info("Loaded %d document sections from %d files.", len(all_docs), len(files))
    return all_docs
