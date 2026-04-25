"""Terminal-based vector-metadata mutation for the Chroma library.

Adapted from the Gemini-authored triage prototype. Differences:

- Aggregates records by stable `doc_id` (set at ingestion) rather than by
  `os.path.basename(source)`. Source is a human label, not a key; SEC-ripped
  filings carry a label like "SEC EDGAR 10-K: AAPL" which breaks basename
  grouping.
- Writes the authoritative `category` field plus the derived
  `temporal_validity` axis in a single call, via `rag.set_category`, so
  mutation logic stays in one place and can never drift from the UI path.
- Skips embedding entirely — metadata-only update on existing IDs.

Invoke from the repo root:

    python3 migrate_library_categories.py
"""
from __future__ import annotations

import sys

from rag import CATEGORIES, TEMPORAL_VALIDITY, list_documents, set_category


_CHOICE_TO_CATEGORY = {
    "1": "textbook",
    "2": "market_report",
    "3": "sec_filing",
    "4": "uncategorized",
}


def _prompt_for_category(source: str, doc_id: str, current: str) -> str | None:
    print()
    print(f"Document : {source}")
    print(f"doc_id   : {doc_id}")
    print(f"current  : {current}  ({TEMPORAL_VALIDITY.get(current, 'unknown')})")
    print("  1) textbook       (perennial)")
    print("  2) market_report  (ephemeral)")
    print("  3) sec_filing     (ephemeral)")
    print("  4) uncategorized  (unknown)")
    print("  s) skip this document")
    print("  q) quit")
    raw = input("Select [1/2/3/4/s/q]: ").strip().lower()
    if raw in ("q", "quit", "exit"):
        return "__quit__"
    if raw in ("s", "skip", ""):
        return None
    if raw in _CHOICE_TO_CATEGORY:
        return _CHOICE_TO_CATEGORY[raw]
    print(f"  ! invalid input {raw!r}; skipping.")
    return None


def run_metadata_migration() -> None:
    print("Initiating Vector Mutation Sequence...")
    docs = list_documents()
    if not docs:
        print("Library is empty. Halting.")
        return

    print(f"Found {len(docs)} document(s) in the library.")

    updates: list[tuple[str, str]] = []   # (doc_id, new_category)
    for d in docs:
        choice = _prompt_for_category(d["source"], d["doc_id"], d["category"])
        if choice == "__quit__":
            print("User aborted. No further prompts.")
            break
        if choice is None:
            continue
        if choice == d["category"]:
            print(f"  = unchanged ({choice})")
            continue
        updates.append((d["doc_id"], choice))

    if not updates:
        print("\nNothing to mutate. Exiting.")
        return

    print(f"\nApplying {len(updates)} mutation(s)...")
    total_chunks = 0
    for doc_id, new_cat in updates:
        n = set_category(doc_id, new_cat)
        temporal = TEMPORAL_VALIDITY[new_cat]
        print(
            f"  - {doc_id:<60}  -> {new_cat:<14} ({temporal})  "
            f"[{n} chunks]"
        )
        total_chunks += n

    print(f"\nMigration complete. {total_chunks} vectors updated across "
          f"{len(updates)} document(s).")
    print(f"Known categories: {list(CATEGORIES.keys())}")


if __name__ == "__main__":
    try:
        run_metadata_migration()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
