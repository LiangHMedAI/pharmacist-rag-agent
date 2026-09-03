import glob
import hashlib
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

from config import (
    EMBEDDING_MODEL,
    KB_PATH,
    RAG_MAX_DISTANCE,
    RAG_TOP_K,
    UPLOAD_MAX_BYTES,
)


@dataclass(frozen=True)
class SearchResult:
    document: object
    distance: float

    @property
    def display_score(self):
        return round(1 / (1 + max(self.distance, 0)), 3)


def get_kb_hash():
    pdf_files = sorted(glob.glob(str(KB_PATH / "*.pdf")))
    if not pdf_files:
        return "empty"
    digest = hashlib.sha256()
    for filename in pdf_files:
        digest.update(filename.encode("utf-8"))
        digest.update(str(os.path.getmtime(filename)).encode("utf-8"))
    return digest.hexdigest()


@st.cache_resource
def load_knowledge_base(_kb_hash):
    """Build a normalized FAISS index from the current PDF collection."""
    pdf_files = glob.glob(str(KB_PATH / "*.pdf"))
    if not pdf_files:
        return None

    documents = []
    for pdf_path in pdf_files:
        try:
            documents.extend(PyPDFLoader(pdf_path).load())
        except Exception:
            continue
    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, embeddings)


def search_knowledge_base(query, k=RAG_TOP_K, max_distance=RAG_MAX_DISTANCE):
    vector_db = load_knowledge_base(get_kb_hash())
    if vector_db is None:
        return []
    matches = vector_db.similarity_search_with_score(query, k=k)
    return [
        SearchResult(document=doc, distance=float(distance))
        for doc, distance in matches
        if float(distance) <= max_distance
    ]


def extract_pdf_text(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip()


def save_uploaded_pdf(filename, pdf_bytes):
    """Validate and save a PDF under a sanitized, content-addressed filename."""
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("The uploaded file is not a valid PDF.")
    if len(pdf_bytes) > UPLOAD_MAX_BYTES:
        raise ValueError("The PDF exceeds the configured upload size limit.")

    extracted_text = extract_pdf_text(pdf_bytes)
    if len(extracted_text) < 30:
        raise ValueError(
            "No usable text was detected. This may be a scanned PDF that requires OCR."
        )

    KB_PATH.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem).strip("._") or "document"
    digest = hashlib.sha256(pdf_bytes).hexdigest()[:10]
    target = KB_PATH / f"{stem}-{digest}.pdf"
    target.write_bytes(pdf_bytes)
    load_knowledge_base.clear()
    return target, extracted_text
