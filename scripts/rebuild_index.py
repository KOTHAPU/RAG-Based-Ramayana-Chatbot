"""
rebuild_index.py — Drop and rebuild the ChromaDB index from scratch.

This is a convenience wrapper around ingest_documents.py --rebuild.

Usage:
    python scripts/rebuild_index.py [--data-dir PATH]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingest_documents import main as ingest_main
import sys


def main() -> None:
    # Inject --rebuild flag and delegate to ingest_main
    if "--rebuild" not in sys.argv:
        sys.argv.append("--rebuild")
    sys.exit(ingest_main())


if __name__ == "__main__":
    main()
