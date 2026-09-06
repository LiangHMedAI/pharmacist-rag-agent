"""
rag_core.py - lightweight retrieval for Render free tier (512 MB RAM).

This module deliberately drops the heavy runtime imports: LangChain document
loaders / vectorstores / text splitters, HuggingFaceEmbeddings,
sentence-transformers, PyTorch and FAISS. PDF parsing relies on PyPDF2 only;
retrieval is a pure-Python character / word n-gram cosine similarity computed
over ~700-character overlapping chunks. Nothing that allocates hundreds of MB
is imported at runtime.

Public interface is kept byte-for-byte compatible so app.py and
llm_service.py do not change:

    SearchResult
    get_kb_hash()
    load_knowledge_base()
    search_knowledge_base()
    extract_pdf_text()
    save_uploaded_pdf()

Search hits keep the LangChain-ish shape used by the rest of the app: each
document exposes .page_content (a string) and .metadata (a dict with 'source'
and 'page'). llm_service.source_name() turns the source into a safe display
filename and page_number() renders a human-friendly page, unchanged.
"""

import glob
import hashlib
import io
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st
from PyPDF2 import PdfReader

from config import KB_PATH, RAG_MAX_DISTANCE, RAG_TOP_K, UPLOAD_MAX_BYTES

# ---------------------------------------------------------------------------
# Chunking parameters (kept close to the original 700/100 values)
# ---------------------------------------------------------------------------
_CHUNK_SIZE = 700
_CHUNK_OVERLAP = 100
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "]


@dataclass(frozen=True)
class SearchResult:
    document: object
    distance: float

    @property
    def display_score(self):
        return round(1 / (1 + max(self.distance, 0)), 3)


@dataclass
class SourceDocument:
    """LangChain-compatible document stand-in: page_content + metadata."""

    page_content: str
    metadata: dict = field(default_factory=dict)


def get_kb_hash():
    pdf_files = sorted(glob.glob(str(KB_PATH / "*.pdf")))
    if not pdf_files:
        return "empty"
    digest = hashlib.sha256()
    for filename in pdf_files:
        digest.update(filename.encode("utf-8"))
        digest.update(str(os.path.getmtime(filename)).encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Character / word n-gram tokenization for a lightweight score
# ---------------------------------------------------------------------------
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokens(text):
    """Return a flat token list used by the cosine bag-of-tokens.

    Latin words are tokenized as whole lowercase words; CJK text is tokenized
    into overlapping character bigrams (plus single chars) so a Chinese query
    still matches Chinese chunks without any embedding model.
    """
    if not text:
        return []
    text = text.lower()
    tokens = _WORD_RE.findall(text)  # latin words / digits
    cjk_run = "".join(_CJK_RE.findall(text))
    for i in range(len(cjk_run) - 1):
        tokens.append(cjk_run[i : i + 2])  # CJK bigrams
    if cjk_run:
        tokens.extend(list(cjk_run))  # single CJK chars (1-char queries)
    # keep meaningful tokens only
    return [t for t in tokens if len(t) >= 1]


def _counter(tokens):
    return Counter(tokens)


def _cosine(a_tokens, b_tokens):
    ca, cb = _counter(a_tokens), _counter(b_tokens)
    if not ca or not cb:
        return 0.0
    inter = sum((ca & cb).values())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return inter / (na * nb)


# ---------------------------------------------------------------------------
# Recursive character chunking (keeps ~700-char chunks with ~100 overlap)
# ---------------------------------------------------------------------------
def _split_text(text, separators=None, chunk_size=_CHUNK_SIZE, overlap=_CHUNK_OVERLAP):
    seps = separators if separators is not None else _SEPARATORS
    chunks = []
    remaining = (text or "").strip()
    while len(remaining) > chunk_size:
        cut = _cut_point(remaining, seps, chunk_size)
        piece = remaining[:cut].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[cut:]
    tail = remaining.strip()
    if tail:
        chunks.append(tail)

    # Carry the tail of each chunk into the next so boundaries keep context.
    if overlap > 0 and len(chunks) > 1:
        merged = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = merged[-1]
            tail_prev = prev[-overlap:] if len(prev) > overlap else prev
            merged.append((tail_prev + chunks[i]).strip())
        chunks = merged
    return [c for c in chunks if c]


def _cut_point(text, separators, chunk_size):
    if len(text) <= chunk_size:
        return len(text)
    window = text[:chunk_size]
    for sep in separators:
        if not sep:
            continue
        idx = window.rfind(sep)
        if idx > max(int(chunk_size * 0.5), len(sep)):
            return idx + len(sep)
    return chunk_size


def _extract_pages(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        value = page.extract_text()
        if value is None:
            value = str()
        pages.append(value)
    return pages


def _build_documents(pdf_files):
    documents = []
    for pdf_path in sorted(pdf_files):
        try:
            pages = _extract_pages(pdf_path)
        except Exception:
            continue
        source = os.path.basename(pdf_path)
        for page_no, page_text in enumerate(pages):
            if not (page_text or str()).strip():
                continue
            for chunk in _split_text(page_text):
                meta = {"source": source, "page": page_no}
                documents.append(_mk_doc(chunk, meta))
    return documents


def _mk_doc(text, meta):
    return SourceDocument(page_content=text, metadata=meta)


def load_knowledge_base(_kb_hash):
    # Rebuild from the current PDF collection whenever the hash changes.
    # Returns a normalized list of SourceDocument chunks, or None if empty.
    return _load_documents()


def _load_documents():
    pdf_files = glob.glob(str(KB_PATH / "*.pdf"))
    if not pdf_files:
        return None
    documents = _build_documents(pdf_files)
    if not documents:
        return None
    return documents


def search_knowledge_base(query, k=RAG_TOP_K, max_distance=RAG_MAX_DISTANCE):
    q = (query or "").strip()
    if not q:
        return []
    qtokens = _tokens(q)
    if not qtokens:
        return []

    docs = load_knowledge_base(get_kb_hash())
    if not docs:
        return []

    scored = []
    for doc in docs:
        dtokens = _tokens(doc.page_content)
        sim = _cosine(qtokens, dtokens)
        dist = 1.0 - sim  # cosine distance in [0, 1]; 1.0 == no lexical overlap
        scored.append((dist, doc))

    scored.sort(key=lambda pair: pair[0])
    results = []
    for dist, doc in scored[:k]:
        # Fail closed: a hit with dist == 1.0 shares no token with the query and
        # must not be presented as evidence. Each returned result must also be
        # inside the caller-supplied max_distance ceiling (default keeps
        # compatibility; cosine distance is bounded above at 1.0 anyway).
        if dist < 1.0 and dist <= max_distance:
            results.append(SearchResult(document=doc, distance=float(dist)))
    return results


def extract_pdf_text(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts = []
    for page in reader.pages:
        value = page.extract_text()
        text_parts.append(value or str())
    return "\n".join(text_parts).strip()


def save_uploaded_pdf(filename, pdf_bytes):
    # Validate signature, size and extractable text before storing a PDF.
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
    target = KB_PATH / (stem + "-" + digest + ".pdf")
    target.write_bytes(pdf_bytes)
    load_knowledge_base.clear()
    return target, extracted_text
