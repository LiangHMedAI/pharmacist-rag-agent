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


# Drug-alias -> source filename map. This is identity metadata only; it never
# injects medical claims, it just steers retrieval toward the right package
# insert when a query plainly names a drug. Lowercase aliases are matched
# against the lowercased query.
DRUG_SOURCE_ALIASES = [
    (("aspirin", "阿司匹林", "acetylsalicylic"), "Aspirin_Package_Insert.pdf"),
    (("ibuprofen", "布洛芬"), "Ibuprofen_Package_Insert.pdf"),
    (("clopidogrel", "氯吡格雷"), "Clopidogrel_Tablets_Insert.pdf"),
    (("eguiyangxue", "阿归养血", "阿归养血颗粒"), "EguiYangxue_Package_Insert.pdf"),
]

# Cosine-distance bonus given to chunks that belong to a source file whose name
# matches a drug named in the query (~0.5 distance means a strong preference,
# but does not by itself let a zero-overlap chunk pass the fail-closed gate).
_MATCHED_SOURCE_BONUS = 0.6


def _named_source_filenames(query):
    """Return {source_filename} whose drug alias appears in the (lowercased) query."""
    q = (query or "").lower()
    matched = set()
    for aliases, filename in DRUG_SOURCE_ALIASES:
        if any(a.lower() in q for a in aliases):
            matched.add(filename)
    return matched


def _rank_documents(qtokens, docs, named_sources, k, max_distance):
    """Rank document chunks; prefer chunks from a source whose filename matches
    a drug named in the query, but stay fail-closed (return no result when a
    chunk shares no token with the query). Returns list[SearchResult]."""
    scored = []
    for doc in docs:
        dtokens = _tokens(doc.page_content)
        sim = _cosine(qtokens, dtokens)
        dist = 1.0 - sim  # 1.0 == no lexical overlap (fail-closed below)
        source = (doc.metadata.get("source") or "") if doc.metadata else ""
        source_name_only = os.path.basename(str(source))
        rank_dist = dist
        if named_sources and source_name_only in named_sources:
            rank_dist = max(0.0, dist - _MATCHED_SOURCE_BONUS)
        scored.append((dist, rank_dist, doc))

    # Sort by the source-prioritised distance, then raw distance as tiebreak.
    scored.sort(key=lambda item: (item[1], item[0]))
    results = []
    for dist, _rank, doc in scored[:k]:
        # Fail closed: no lexical overlap (dist == 1.0) is never evidence; the
        # caller max_distance ceiling is honoured as a further gate.
        if dist < 1.0 and dist <= max_distance:
            results.append(SearchResult(document=doc, distance=float(dist)))
    return results


def _literal_coverage(qtokens, text):
    """How many distinct query tokens appear (as substrings, lowercased) in text.

    Substring matching is tolerant of PDF text where spaces/segmentation are
    lost (e.g. "IbuprofenTablet" still contains "ibuprofen") and lets a latin
    drug token confirm a chunk even inside a Chinese-language question.
    """
    t = (text or "").lower()
    if not t:
        return 0
    seen = set()
    for tok in qtokens:
        if tok and tok in t:
            seen.add(tok)
    return len(seen)


def search_knowledge_base(query, k=RAG_TOP_K, max_distance=RAG_MAX_DISTANCE):
    """Return evidence chunks for a user query.

    When the query names a known drug that exists in the KB, retrieve only from
    that drug's own package insert and rank chunks by literal (substring) query
    coverage. The confirm path deliberately does NOT require Chinese question
    words to overlap English insert text: once a drug is confirmed present and
    the KB holds it, we return its best section instead of falsely reporting
    "no evidence". Free-form queries (no known drug) use fail-closed cosine.
    """
    q = (query or "").strip()
    if not q:
        return []
    qtokens = _tokens(q)
    if not qtokens:
        return []

    docs = load_knowledge_base(get_kb_hash())
    if not docs:
        return []

    return _search_docs_from_text(q, qtokens, docs, k, max_distance)


def _search_docs_from_text(q, qtokens, docs, k, max_distance):
    """Rank `docs` for a query (exposed as a pure function for tests)."""
    named_sources = _named_source_filenames(q)
    available = {os.path.basename(str(d.metadata.get("source") or "")) for d in docs}
    present = sorted(s for s in named_sources if s in available)

    if present:
        scope = [d for d in docs
                 if os.path.basename(str(d.metadata.get("source") or "")) in present]
        covered = sorted(
            ((_literal_coverage(qtokens, d.page_content), d) for d in scope),
            key=lambda pair: pair[0], reverse=True,
        )
        # Evidence = genuinely overlapping chunks; if none, still surface the
        # matched PDF's top page-ordered section (drug confirmed present).
        picks = [c for c in covered if c[0] > 0] or covered
        maxcov = picks[0][0] if picks else 0
        out = []
        for cov, doc in picks[:k]:
            dist = (1.0 - cov / maxcov) if maxcov else 0.5
            out.append(SearchResult(document=doc, distance=float(dist)))
        return out

    # Free-form: no known drug -> plain fail-closed cosine ranking.
    return _rank_documents(qtokens, docs, set(), k, max_distance)


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
