"""RAG pipeline: cached embedding engine, persistent Chroma store, dedup ingestion.

Reasoning features layered on top of the base pipeline:
  * category + temporal_validity metadata (filter)
  * topic multi-label metadata  (filter)
  * MMR retrieval               (diversity)
  * multi-query decomposition   (coverage for compound questions)
  * question router             (auto filter suggestion)
  * citation-grounded answering (hallucination control)
"""
from __future__ import annotations

import json
import os
import re

import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = "./chroma_db"

# Model used for small utility calls (routing, decomposition). Kept separate
# from the Oracle's answering model so you can tune one without the other.
UTILITY_MODEL = os.getenv("NOODLE_UTILITY_MODEL", "llama3.2")

_SEC_SOURCE_RE = re.compile(r"^SEC EDGAR\s+(?P<form>10-K|8-K)\s*:\s*(?P<ticker>[A-Z0-9.\-]+)")


def _derive_doc_id_from_source(source: str) -> str:
    """Stable key for legacy chunks that lack a written `doc_id` field."""
    if not source:
        return "pdf::unknown"
    m = _SEC_SOURCE_RE.match(source.strip())
    if m:
        return f"sec::{m.group('ticker')}::{m.group('form')}"
    return f"pdf::{os.path.basename(source)}"

# Category taxonomy. Stored on every chunk's metadata so the Oracle can filter.
CATEGORIES = {
    "textbook": "📘 Financial Investing Textbooks",
    "market_report": "🏦 Institutional Market Reports",
    "sec_filing": "🏛️ SEC Filings",
    "uncategorized": "❔ Uncategorized",
}

# Deterministic second axis derived from category. Kept in sync by
# `ingest_chunks` and `set_category` — do not write this field directly.
TEMPORAL_VALIDITY = {
    "textbook": "perennial",
    "market_report": "ephemeral",
    "sec_filing": "ephemeral",
    "uncategorized": "unknown",
}

# Topic vocabulary — a document can carry MANY of these simultaneously.
# Stored as boolean metadata columns (topic_<key>: True/False) because
# Chroma metadata values must be scalars.
TOPIC_LABELS = {
    "valuation":       "💰 Valuation & DCF",
    "macro":           "🌍 Macro & Cycles",
    "credit":          "🏦 Credit & Spreads",
    "accounting":      "📊 Accounting & Earnings Quality",
    "behavioral":      "🧠 Behavioral / Psychology",
    "technicals":      "📈 Technical Analysis",
    "derivatives":     "🎲 Derivatives & Options",
    "shorting":        "📉 Shorting & Activism",
    "m_and_a":         "🤝 M&A / Corporate Actions",
    "fixed_income":    "📜 Fixed Income",
    "geopolitics":     "🗺️ Geopolitics",
    "risk_management": "🛡️ Risk Management",
    "quantitative":    "🤖 Quant / ML",
    "value_investing": "🛡️ Value Investing",
    "growth_investing":"🚀 Growth Investing",
    "corporate_finance":"💼 Corporate Finance",
}
TOPICS = list(TOPIC_LABELS.keys())
TOPIC_META_PREFIX = "topic_"


def topic_key(topic: str) -> str:
    return f"{TOPIC_META_PREFIX}{topic}"


def topics_from_metadata(md: dict) -> list[str]:
    """Read the set of True topic_* booleans from a chunk metadata dict."""
    if not md:
        return []
    return [t for t in TOPICS if md.get(topic_key(t)) is True]


@st.cache_resource
def embedding_engine():
    return OllamaEmbeddings(model="nomic-embed-text")


@st.cache_resource
def vector_db():
    return Chroma(persist_directory=DB_DIR, embedding_function=embedding_engine())


def already_ingested(doc_id: str) -> bool:
    try:
        existing = vector_db().get(where={"doc_id": doc_id}, limit=1)
        return bool(existing and existing.get("ids"))
    except Exception:
        return False


def ingest_chunks(
    chunks,
    doc_id: str,
    source_label: str,
    category: str = "uncategorized",
    topics: list[str] | None = None,
) -> int:
    if category not in CATEGORIES:
        category = "uncategorized"
    temporal = TEMPORAL_VALIDITY[category]
    topics = [t for t in (topics or []) if t in TOPICS]
    for c in chunks:
        c.metadata["doc_id"] = doc_id
        c.metadata["source"] = source_label
        c.metadata["category"] = category
        c.metadata["temporal_validity"] = temporal
        # Write every topic column so filters are consistent — False where absent.
        for t in TOPICS:
            c.metadata[topic_key(t)] = t in topics
    vector_db().add_documents(chunks)
    return len(chunks)


def list_documents() -> list[dict]:
    """Return one entry per unique doc_id in the library.

    Each entry: {doc_id, source, category, chunks}.
    """
    try:
        raw = vector_db().get(include=["metadatas"])
    except Exception:
        return []
    metadatas = raw.get("metadatas") or []
    by_id: dict[str, dict] = {}
    for md in metadatas:
        if not md:
            continue
        # Legacy chunks (pre-category schema) have no doc_id. Fall back to a
        # deterministic key derived from source so they still appear in the
        # Manage Library UI — the migration script will backfill doc_id properly.
        doc_id = md.get("doc_id") or _derive_doc_id_from_source(md.get("source", ""))
        if not doc_id:
            continue
        if doc_id not in by_id:
            cat = md.get("category", "uncategorized")
            by_id[doc_id] = {
                "doc_id": doc_id,
                "source": md.get("source", doc_id),
                "category": cat,
                "temporal_validity": md.get(
                    "temporal_validity", TEMPORAL_VALIDITY.get(cat, "unknown")
                ),
                "topics": topics_from_metadata(md),
                "chunks": 0,
            }
        by_id[doc_id]["chunks"] += 1
    return sorted(by_id.values(), key=lambda d: d["source"].lower())


def set_category(doc_id: str, category: str) -> int:
    """Re-label every chunk belonging to `doc_id`. Returns count updated.

    Supports legacy chunks that have no written `doc_id` field by falling
    back to a source-derived key. When matched that way, this call also
    *backfills* the real `doc_id` onto each chunk — converting legacy
    records into first-class library entries.
    """
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}")
    db = vector_db()
    ids, metadatas = _resolve_chunks_for_doc(db, doc_id)
    if not ids:
        return 0
    temporal = TEMPORAL_VALIDITY[category]
    new_metas = []
    for md in metadatas:
        md = dict(md or {})
        md["doc_id"] = doc_id           # backfill for legacy rows
        md["category"] = category
        md["temporal_validity"] = temporal
        new_metas.append(md)
    db._collection.update(ids=ids, metadatas=new_metas)
    return len(ids)


def _resolve_chunks_for_doc(db, doc_id: str):
    """Return (ids, metadatas) for every chunk in `doc_id`, with legacy fallback.

    Shared by set_category / set_topics / delete_document so mutation logic
    cannot drift.
    """
    try:
        raw = db.get(where={"doc_id": doc_id}, include=["metadatas"])
    except Exception:
        raw = {"ids": [], "metadatas": []}
    ids = list(raw.get("ids") or [])
    metadatas = list(raw.get("metadatas") or [])
    if ids:
        return ids, metadatas

    try:
        all_raw = db.get(include=["metadatas"])
    except Exception:
        return [], []
    all_ids = all_raw.get("ids") or []
    all_metas = all_raw.get("metadatas") or []
    for rid, md in zip(all_ids, all_metas):
        md = md or {}
        if md.get("doc_id"):
            continue
        if _derive_doc_id_from_source(md.get("source", "")) == doc_id:
            ids.append(rid)
            metadatas.append(md)
    return ids, metadatas


def set_topics(doc_id: str, topics: list[str]) -> int:
    """Replace the full topic set on every chunk in `doc_id`.

    Writes a boolean for every known topic (True for the ones in `topics`,
    False for the rest) so metadata filtering is consistent across the library.
    """
    cleaned = [t for t in (topics or []) if t in TOPICS]
    db = vector_db()
    ids, metadatas = _resolve_chunks_for_doc(db, doc_id)
    if not ids:
        return 0
    new_metas = []
    for md in metadatas:
        md = dict(md or {})
        md["doc_id"] = doc_id  # backfill for legacy rows
        for t in TOPICS:
            md[topic_key(t)] = t in cleaned
        new_metas.append(md)
    db._collection.update(ids=ids, metadatas=new_metas)
    return len(ids)


def delete_document(doc_id: str) -> int:
    """Remove every chunk belonging to `doc_id`. Returns count removed."""
    db = vector_db()
    ids, _ = _resolve_chunks_for_doc(db, doc_id)
    if not ids:
        return 0
    db._collection.delete(ids=ids)
    return len(ids)


# ---------------------------------------------------------------------------
# Retrieval + reasoning layer (MMR, multi-query, router, citation grounding)
# ---------------------------------------------------------------------------


def compose_filter(
    categories: list[str] | None = None,
    topics_any: list[str] | None = None,
    ticker: str | None = None,
) -> dict | None:
    """Build a Chroma `where` dict from the three supported filter axes.

    - categories: match if chunk's `category` is in this list.
    - topics_any: match if ANY of the listed topic booleans is True (OR).
    - ticker:     match if chunk's `ticker` equals this (SEC chunks carry it).

    Returns None when no filter is required.
    """
    clauses: list[dict] = []

    if categories:
        all_cats = list(CATEGORIES.keys())
        if 0 < len(categories) < len(all_cats):
            clauses.append(
                {"category": categories[0]}
                if len(categories) == 1
                else {"category": {"$in": list(categories)}}
            )

    if topics_any:
        topic_clauses = [{topic_key(t): True} for t in topics_any if t in TOPICS]
        if len(topic_clauses) == 1:
            clauses.append(topic_clauses[0])
        elif len(topic_clauses) > 1:
            clauses.append({"$or": topic_clauses})

    if ticker:
        clauses.append({"ticker": ticker})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def retrieve(
    query: str,
    k: int = 6,
    categories: list[str] | None = None,
    topics_any: list[str] | None = None,
    ticker: str | None = None,
    use_mmr: bool = True,
    fetch_k: int = 30,
    mmr_lambda: float = 0.5,
):
    """Run a single retrieval with MMR (default) or plain similarity.

    Falls back gracefully: if the filter yields nothing, retries without the
    ticker clause; if that still yields nothing, drops all filters.
    """
    db = vector_db()

    def _search(flt):
        try:
            if use_mmr:
                return db.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k, lambda_mult=mmr_lambda, filter=flt
                )
            return db.similarity_search(query, k=k, filter=flt)
        except Exception:
            return []

    # Try the full filter.
    full_filter = compose_filter(categories, topics_any, ticker)
    docs = _search(full_filter) if full_filter else _search(None)
    if docs:
        return docs

    # Retry without ticker.
    if ticker:
        fallback = compose_filter(categories, topics_any, None)
        docs = _search(fallback) if fallback else _search(None)
        if docs:
            return docs

    # Last resort: unfiltered.
    return _search(None)


def retrieve_multi(
    queries: list[str],
    k_per_query: int = 4,
    k_total: int = 8,
    **retrieve_kwargs,
):
    """Run retrieval for a list of sub-queries, dedupe, return top-k_total."""
    seen: dict[str, object] = {}
    order: list[str] = []
    for q in queries:
        for doc in retrieve(q, k=k_per_query, **retrieve_kwargs):
            key = doc.page_content[:200]
            if key in seen:
                continue
            seen[key] = doc
            order.append(key)
            if len(order) >= k_total:
                break
        if len(order) >= k_total:
            break
    return [seen[k] for k in order]


def _ollama_json(system: str, user: str, model: str | None = None) -> dict | None:
    """Utility: call Ollama and parse the first {...} JSON blob in the output."""
    try:
        import ollama  # lazy import — only needed for reasoning features
    except Exception:
        return None
    try:
        resp = ollama.chat(
            model=model or UTILITY_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format="json",
        )
        text = resp["message"]["content"].strip()
    except Exception:
        return None
    # Best-effort parse — format="json" usually gives clean JSON, but guard anyway.
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def decompose_query(query: str, max_sub: int = 3) -> list[str]:
    """Break a compound question into up to `max_sub` focused sub-queries.

    Returns `[query]` if decomposition fails or the question is simple.
    """
    if not query or len(query.split()) < 8:
        return [query]
    system = (
        "You decompose financial-analysis questions into focused sub-questions "
        "that improve vector retrieval. Return JSON ONLY in the shape "
        '{"sub_queries": ["...", "..."]}. Each sub-query should be independently '
        "searchable, ≤ 15 words, and together they should cover the original "
        "question. If the question is simple, return a single-item list."
    )
    user = f"Original question:\n{query}\n\nmax_sub_queries = {max_sub}"
    data = _ollama_json(system, user)
    if not data:
        return [query]
    subs = data.get("sub_queries") or []
    subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
    if not subs:
        return [query]
    # Always keep the original as the first query so we never lose the intent.
    out = [query] + [s for s in subs if s.lower() != query.lower()]
    return out[: max_sub + 1]


def route_query(query: str) -> dict:
    """Classify a question → suggested {categories, topics, domain}.

    Returns a dict even on failure: {"categories": [], "topics": [],
    "domain": "general", "rationale": ""}.
    """
    fallback = {"categories": [], "topics": [], "domain": "general", "rationale": ""}
    if not query.strip():
        return fallback

    cats_json = json.dumps(list(CATEGORIES.keys()))
    topics_json = json.dumps(TOPICS)
    system = (
        "You route financial-analysis questions to the right document sources. "
        f"Valid categories: {cats_json}. "
        f"Valid topics: {topics_json}. "
        "Return JSON ONLY in the shape "
        '{"categories": ["..."], "topics": ["..."], '
        '"domain": "valuation|macro|accounting|behavioral|legal|mgmt_quality|general", '
        '"rationale": "<one sentence>"}. '
        "Only include categories/topics that are clearly relevant. Prefer precision "
        "over recall — an empty list is fine when unsure."
    )
    data = _ollama_json(system, f"Question:\n{query}")
    if not data:
        return fallback
    cats = [c for c in (data.get("categories") or []) if c in CATEGORIES]
    tops = [t for t in (data.get("topics") or []) if t in TOPICS]
    return {
        "categories": cats,
        "topics": tops,
        "domain": str(data.get("domain") or "general"),
        "rationale": str(data.get("rationale") or "")[:300],
    }


# ----------------------- Citation grounding --------------------------------


_CITATION_RE = re.compile(r"\[chunk_(\d+)\]")


def format_chunks_for_citation(docs) -> str:
    """Render retrieved chunks as `[chunk_N]` blocks for citation prompting."""
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        parts.append(f"[chunk_{i}] source={src}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


# ----------------------- Conflict detection --------------------------------


def detect_conflicts(docs, question: str | None = None) -> list[dict]:
    """Identify pairs of retrieved chunks that contain contradicting claims.

    Returns a list of {a, b, conflict, resolution_hint} dicts where a/b are
    1-based chunk indexes. Uses Ollama with JSON-structured output; returns
    [] on any failure.

    Leverages the temporal_validity axis: conflicts between a `perennial`
    source (textbook principle) and an `ephemeral` one (market report) are
    flagged with a resolution_hint that prefers the perennial source for
    principles and the ephemeral source for present-day facts.
    """
    if not docs or len(docs) < 2:
        return []

    # Compact each chunk: first ~400 chars is enough for contradiction detection.
    lines = []
    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        src = os.path.basename(md.get("source", "?"))
        tv = md.get("temporal_validity", "unknown")
        cat = md.get("category", "unknown")
        excerpt = d.page_content.strip().replace("\n", " ")[:400]
        lines.append(f"[chunk_{i}] ({cat}, {tv}) {src}\n{excerpt}")
    corpus = "\n\n".join(lines)

    system = (
        "You identify CONTRADICTIONS between financial document excerpts. "
        "A contradiction = two chunks making incompatible claims on the SAME "
        "proposition (valuation, rate direction, credit outlook, risk, etc.). "
        "Different scopes or different subjects are NOT contradictions. "
        "Return JSON ONLY in the shape "
        '{"conflicts": [{"a": <int>, "b": <int>, '
        '"conflict": "<one sentence>", "resolution_hint": "<one sentence>"}]}. '
        "Use the (perennial) vs (ephemeral) tags: perennial sources state timeless "
        "principles; ephemeral sources state time-bound facts. When they clash, "
        "the resolution_hint should say which source governs which aspect. "
        "Empty list is the correct answer when chunks merely differ in emphasis."
    )
    user = (
        f"QUESTION: {question or '(general analysis)'}\n\n"
        f"CHUNKS:\n{corpus}\n\n"
        "List every genuine contradiction (empty list if none)."
    )
    data = _ollama_json(system, user)
    if not data:
        return []
    out = []
    n = len(docs)
    for c in data.get("conflicts") or []:
        try:
            a = int(c.get("a"))
            b = int(c.get("b"))
        except (TypeError, ValueError):
            continue
        if not (1 <= a <= n and 1 <= b <= n and a != b):
            continue
        out.append({
            "a": a,
            "b": b,
            "conflict": str(c.get("conflict") or "")[:400],
            "resolution_hint": str(c.get("resolution_hint") or "")[:400],
        })
    return out


def format_conflicts_for_prompt(conflicts: list[dict]) -> str:
    """Render detected conflicts as a CONFLICT DIAGNOSTIC block for the Oracle."""
    if not conflicts:
        return ""
    lines = ["CONFLICT DIAGNOSTIC (resolve these explicitly in your answer):"]
    for c in conflicts:
        lines.append(
            f"- [chunk_{c['a']}] vs [chunk_{c['b']}]: {c['conflict']} "
            f"— hint: {c['resolution_hint']}"
        )
    return "\n".join(lines) + "\n"


def verify_citations(answer: str, docs) -> dict:
    """Check every `[chunk_N]` citation in `answer` against `docs`.

    Returns: {"cited": [int], "unknown": [int], "uncited_indexes": [int]}.
    """
    cited_raw = [int(m.group(1)) for m in _CITATION_RE.finditer(answer)]
    valid_idxs = set(range(1, len(docs) + 1))
    cited = sorted({c for c in cited_raw if c in valid_idxs})
    unknown = sorted({c for c in cited_raw if c not in valid_idxs})
    uncited = sorted(valid_idxs - set(cited))
    return {"cited": cited, "unknown": unknown, "uncited_indexes": uncited}
