"""RAG pipeline: cached embedding engine, persistent Chroma store, dedup ingestion."""
from __future__ import annotations

import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DB_DIR = "./chroma_db"


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


def ingest_chunks(chunks, doc_id: str, source_label: str) -> int:
    for c in chunks:
        c.metadata["doc_id"] = doc_id
        c.metadata["source"] = source_label
    vector_db().add_documents(chunks)
    return len(chunks)
