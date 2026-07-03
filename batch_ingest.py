#!/usr/bin/env python3
"""Batch-ingest PDFs from /Desktop/Hobbies/Finance/PDFS into ChromaDB.

Run from the project root:
    python3 batch_ingest.py

Strategy
--------
* Reads PDFs directly from the source tree — no copy to temp_pdfs — so the
  process uses zero extra disk space for the PDF files themselves.
* Deduplicates by base filename: if a basename is already tracked in Chroma,
  the file is skipped.
* Also skips an explicit list of same-book / different-filename duplicates so
  the same content isn't indexed twice under two different doc_ids.
* Skips the 'Bad PDF Files couldn't be read' folder entirely.
* On failure to load/parse a PDF, prints a warning and continues.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SRC_ROOT = Path("/Users/dimitripelletier/Desktop/Hobbies/Finance/PDFS")
DB_DIR   = Path("/Users/dimitripelletier/Desktop/Projects/NoodleStockTracker/chroma_db")

# Folder to skip entirely (labeled as unreadable by user)
SKIP_FOLDER_LOWER = "bad pdf files couldn't be read"

# Base filenames to skip: same book already indexed under a different (cleaner)
# filename. Skipping prevents duplicate chunks that bias RAG retrieval.
# NOTE: use the EXACT Unicode characters from the real filenames (curly
# apostrophes ’, etc.) — straight-apostrophe copies won't match on disk.
SKIP_BASENAMES: set[str] = {
    # ── Same as "Sam Weinstein - Sam Weinstein's Secrets..." (already indexed) ──
    "Stan Weinstein’s Secrets For Profiting in Bull and Bear -- Stan Weinstein -- 1987 -- McGraw-Hill -- 7688210d9429b740556bd8ee8a4e80d3 -- Anna’s Archive.pdf",
    # ── Same as "Tim Lee - The Rise of Carry..." (already indexed) ──
    "The Rise of Carry- The Dangerous Consequences of Volatility -- Tim Lee, Jamie Lee & Kevin Coldiron -- 2019 -- de910d43b3fdc4c90a93aadbc5d1b2af -- Anna’s Archive.pdf",
    # ── Same as "Thornton L. OGlove - Quality of Earnings.pdf" (already indexed) ──
    "OGlove - Quality of Earnings.pdf",
    # ── Same-book _OceanofPDF duplicates of already-indexed clean-name versions ──
    "_OceanofPDF.com_Anatomy_of_the_Bear_-_Russell_Napier.pdf",
    "_OceanofPDF.com_Dead_Companies_Walking_-_Scott_Fearon_and_Jesse_Powell.pdf",
    "_OceanofPDF.com_Geopolitical_Alpha_An_Investment_Framework_for_Predicting_the_Future_-_Marko_Papic.pdf",
    "_OceanofPDF.com_How_to_Make_Money_Selling_Stocks_Short_-_William_Oneil.pdf",
    "_OceanofPDF.com_Martin_Zweig_Winning_on_Wall_Street_-_Martin_Zweig.pdf",
    "_OceanofPDF.com_Narrative_and_Numbers_-_Aswath_Damodaran.pdf",
    "_OceanofPDF.com_The_Alchemy_of_Finance_-_George_Soros.pdf",
    "_OceanofPDF.com_Traders_Guns_Money_-_Satyajit_Das.pdf",
    "_OceanofPDF.com_Damodaran_on_Valuation_-_Aswath_Damodaran.pdf",
    "_OceanofPDF.com_Financial_Shenanigans_-_Howard_M_Schilit__Jeremy_Perler__Yoni_Engelhart.pdf",
    # ── Same book as clean-name PDFs for LLM versions ──
    "_OceanofPDF.com_Liars_Poker_Norton_Paperback_-_Michael_Lewis.pdf",
    "_OceanofPDF.com_One_Up_on_Wall_Street__How_to_Use_What_You_-_Peter_Lynch.pdf",
    "_OceanofPDF.com_The_Fund_-_Rob_Copeland.pdf",
    "_OceanofPDF.com_How_to_Create_Lifetime_CashFlow_Through_Multifamily_Properties_-_Rod_Khleif.pdf",
    # ── Probability book — duplicate "(1)" copy ──
    "_OceanofPDF.com_Introduction_to_Probability_-_Dimitri_Bertsekas_and_John_Tsitsiklis (1).pdf",
    # ── Stochastic processes copy with duplicate content ──
    "Probability, Random Variables And Stochastic Processes 4Th Ed - A Papoulis, S Pillai (Mcgraw-Hill) Ww copy.pdf",
    # ── Italian-language edition of Fooled by Randomness ──
    "Trading_Nassim.N.Taleb Giocati.dal.Caso (Nassim.N.Taleb) (Z-Library).pdf",
    # ── Stefan Jansen ML for Algo Trading already indexed — this is same book ──
    "Machine_Learning_for_Algorithmic_Trading_Predictive.pdf",
    # ── Bibliography document, not a book ──
    "AF_with_Nick_Bibliography.01.pdf",
    # ── Unknown file ──
    "book1.pdf",
    # ── Already indexed under same basename in other subfolders ──
    "Pedersen - Effeciently Inefficient.pdf",   # already in ChromaDB
    "Mark Zweig - Winning on Wall Street.pdf",   # already in ChromaDB
    "Marko Papic - Geopolitical Alpha.pdf",      # already in ChromaDB
    "Russell Napier - Anatomy of the Bear.pdf",  # already in ChromaDB
    "Sam Weinstein - Sam Weinstein's Secrets for Profiting in Bull and Bear Markets.pdf",
    "Satayajit Das - Traders Guns and Money.pdf",
    "Scott Fearon - Dead Companies Walking.pdf",
    "Seth Klarman - Margin of Safety.pdf",
    "Stefan Jansen - Machine Learning for Algorithmic Trading.pdf",
    "The Alchemy of Finance - George Soros.pdf",
    "Thornton L. OGlove - Quality of Earnings.pdf",
    "Tim Lee - The Rise of Carry- The Dangerous Consequences of Volatility.pdf",
    "William ONiel - How to Make Money selling stocks short.pdf",
    "J Voit - The Statistical Mechanics of Financial Markets.pdf",
    "Shiller2.pdf",
    "bii-global-outlook-2026.pdf",
}

# ---------------------------------------------------------------------------
# Minimal direct setup (no Streamlit required)
# ---------------------------------------------------------------------------

def _get_chroma():
    """Return a raw chromadb collection (bypasses LangChain caching)."""
    import chromadb
    return chromadb.PersistentClient(path=str(DB_DIR))


def _get_langchain_vectorstore():
    """Instantiate the same LangChain Chroma wrapper rag.py uses."""
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    emb = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=str(DB_DIR), embedding_function=emb)


def _already_ingested_basenames() -> set[str]:
    """Return the set of base filenames (without directory) already in Chroma."""
    client = _get_chroma()
    try:
        col = client.get_collection("langchain")
    except Exception:
        return set()
    result = col.get(include=["metadatas"])
    basenames: set[str] = set()
    for md in result.get("metadatas") or []:
        src = md.get("source", "")
        if src:
            basenames.add(os.path.basename(src))
    return basenames


def _collect_pdfs() -> list[Path]:
    """Walk SRC_ROOT and return paths of PDFs to ingest."""
    pdfs = []
    for p in sorted(SRC_ROOT.rglob("*.pdf")):
        # Skip the 'Bad PDF Files' folder
        parts_lower = [part.lower() for part in p.parts]
        if any(SKIP_FOLDER_LOWER in part for part in parts_lower):
            continue
        pdfs.append(p)
    return pdfs


def _ingest_pdf(path: Path, vs) -> tuple[int, str]:
    """Load, chunk, and add one PDF to the vector store.

    Returns (chunk_count, error_string). error_string is empty on success.
    """
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    try:
        loader  = PyMuPDFLoader(str(path))
        pages   = loader.load()
    except Exception as e:
        return 0, f"Load error: {e}"

    if not pages:
        return 0, "No pages extracted"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(pages)
    if not chunks:
        return 0, "No chunks after split"

    basename = path.name
    doc_id   = f"pdf::{basename}"
    now_ts   = int(time.time())

    for c in chunks:
        c.metadata["doc_id"]   = doc_id
        c.metadata["source"]   = basename          # matches what rag.py shows in UI
        c.metadata["category"] = "textbook"        # most are textbooks; user can re-label in UI
        c.metadata["temporal_validity"] = "perennial"
        c.metadata["ingested_at"] = now_ts
        # Populate all topic booleans as False (user can tag via Manage Library)
        topic_keys = [
            "valuation", "macro", "credit", "accounting", "behavioral",
            "technicals", "derivatives", "shorting", "m_and_a",
            "fixed_income", "geopolitics", "risk_management",
            "quantitative", "value_investing", "growth_investing",
            "corporate_finance",
        ]
        for t in topic_keys:
            c.metadata[f"topic_{t}"] = False

    try:
        vs.add_documents(chunks)
    except Exception as e:
        return 0, f"Chroma write error: {e}"

    return len(chunks), ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("NoodleStockTracker — Batch PDF Ingest")
    print(f"Source : {SRC_ROOT}")
    print(f"Chroma : {DB_DIR}")
    print("=" * 70)

    # Step 1 — collect and filter
    print("\n[1/4] Scanning source folder…")
    all_pdfs = _collect_pdfs()
    print(f"      Found {len(all_pdfs)} PDF files (excl. 'Bad PDF' folder)")

    print("\n[2/4] Reading existing Chroma index…")
    indexed = _already_ingested_basenames()
    print(f"      {len(indexed)} unique sources already in ChromaDB")

    to_ingest: list[Path] = []
    skipped_indexed = []
    skipped_explicit = []

    for p in all_pdfs:
        bn = p.name
        if bn in indexed:
            skipped_indexed.append(bn)
        elif bn in SKIP_BASENAMES:
            skipped_explicit.append(bn)
        else:
            to_ingest.append(p)

    print(f"\n      Already indexed  : {len(skipped_indexed)} files  (will skip)")
    print(f"      Known duplicates : {len(skipped_explicit)} files  (will skip)")
    print(f"      New to ingest    : {len(to_ingest)} files")

    if not to_ingest:
        print("\n✅  Nothing new to ingest — library is up-to-date.")
        return

    print("\n[3/4] Initialising embeddings (Ollama nomic-embed-text)…")
    vs = _get_langchain_vectorstore()
    print("      Vector store ready.")

    print(f"\n[4/4] Ingesting {len(to_ingest)} PDFs…\n")
    total_chunks = 0
    errors: list[tuple[str, str]] = []
    # Track basenames ingested THIS session so same-named files in different
    # subfolders (e.g. Ray Dalio books duplicated across Market Cycles & PDFs
    # for LLM) are only embedded once.
    session_ingested: set[str] = set()
    skipped_session_dup = []

    for i, path in enumerate(to_ingest, 1):
        short = path.relative_to(SRC_ROOT)
        bn = path.name
        if bn in session_ingested:
            print(f"  [{i:3d}/{len(to_ingest)}] {short} … ⏭  same basename already ingested this run")
            skipped_session_dup.append(str(short))
            continue
        print(f"  [{i:3d}/{len(to_ingest)}] {short} … ", end="", flush=True)
        n, err = _ingest_pdf(path, vs)
        if err:
            print(f"❌  {err}")
            errors.append((str(short), err))
        else:
            total_chunks += n
            session_ingested.add(bn)
            print(f"✅  {n} chunks")

    print("\n" + "=" * 70)
    print(f"Ingestion complete.")
    print(f"  New chunks added   : {total_chunks:,}")
    print(f"  Files succeeded    : {len(session_ingested)}")
    print(f"  Skipped (dup run)  : {len(skipped_session_dup)}")
    print(f"  Files failed       : {len(errors)}")
    if errors:
        print("\nFailed files:")
        for fname, err in errors:
            print(f"  ✗  {fname}")
            print(f"     {err}")
    print("=" * 70)
    print("\nNext steps in the app:")
    print("  • Library → Manage Library → assign topics/categories to new docs")
    print("  • Restart Streamlit if it's running to pick up the new chunks")


if __name__ == "__main__":
    sys.exit(main())
