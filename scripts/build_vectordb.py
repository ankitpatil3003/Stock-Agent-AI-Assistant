# scripts/build_vectordb.py
import os, sys, glob, uuid, gc
from typing import List, Tuple, Dict, Any, Iterable

# Make imports work from anywhere
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config.settings import settings, init_logging

# Prefer PyMuPDF; fallback: pdfminer
try:
    import fitz  # PyMuPDF
    PYMUPDF = True
except Exception:
    PYMUPDF = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    PDFMINER = True
except Exception:
    PDFMINER = False


# --------- helpers ---------

def read_pdf_pages(path: str) -> Iterable[Tuple[int, str]]:
    """
    Yields (page_number, text) one at a time to avoid large memory spikes.
    """
    if PYMUPDF:
        import fitz
        with fitz.open(path) as doc:
            for i in range(len(doc)):
                txt = doc.load_page(i).get_text("text") or ""
                yield (i + 1, txt)
        return
    if PDFMINER:
        # pdfminer returns whole doc; split with form feed.
        # To keep memory sane, iterate chunks immediately.
        text = pdfminer_extract_text(path) or ""
        for i, t in enumerate(text.split("\x0c"), start=1):
            yield (i, t)
        return
    raise RuntimeError("Install PyMuPDF or pdfminer.six for PDF parsing.")


def clean_text(s: str) -> str:
    # Normalize whitespace without duplicating giant strings
    return " ".join(s.split())


def chunk_text_iter(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    """
    Generator: yields chunks without building a list in memory.
    """
    n = len(text)
    if n == 0:
        return
    start = 0
    # Safety: avoid negative or pathological overlap
    overlap = max(0, min(overlap, chunk_size // 2))
    while start < n:
        end = min(n, start + chunk_size)
        yield text[start:end]
        # move window
        if end == n:
            break
        start = end - overlap


def batched(ids, docs, metas, batch_size=64):
    """
    Generator that yields small batches to Chroma.
    """
    cur_ids, cur_docs, cur_metas = [], [], []
    for i, d, m in zip(ids, docs, metas):
        cur_ids.append(i)
        cur_docs.append(d)
        cur_metas.append(m)
        if len(cur_ids) >= batch_size:
            yield cur_ids, cur_docs, cur_metas
            cur_ids, cur_docs, cur_metas = [], [], []
    if cur_ids:
        yield cur_ids, cur_docs, cur_metas


def main():
    init_logging()

    # Tunables (you can also move these to .env if you want)
    CHUNK_SIZE = 900       # smaller chunks reduce RAM spikes
    OVERLAP = 150
    MAX_PAGE_CHARS = 200_000   # hard cap per page to avoid monster pages
    BATCH_SIZE = 64

    os.makedirs(settings.CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=settings.CHROMA_DIR)

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=settings.EMBEDDING_MODEL)
    col = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    pdf_paths = sorted(glob.glob(os.path.join(settings.PDF_DIR, "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {settings.PDF_DIR}")
        return

    for pdf_path in pdf_paths:
        base = os.path.basename(pdf_path)
        print(f"Indexing: {base}")

        total_chunks = 0
        # Process ONE page at a time → stream chunks → add in small batches
        for page_no, raw_text in read_pdf_pages(pdf_path):
            if not raw_text:
                continue
            # Trim crazy pages
            if len(raw_text) > MAX_PAGE_CHARS:
                raw_text = raw_text[:MAX_PAGE_CHARS]

            text = clean_text(raw_text)
            if not text:
                continue

            # Stream chunks; don't collect giant lists
            ids, docs, metas = [], [], []
            for ci, chunk in enumerate(chunk_text_iter(text, CHUNK_SIZE, OVERLAP)):
                _id = f"{base}:{page_no}:{ci}:{uuid.uuid4().hex[:8]}"
                ids.append(_id)
                docs.append(chunk)
                metas.append({"source": base, "page": page_no})

                # Flush in mini-batches to cap memory
                if len(ids) >= BATCH_SIZE:
                    col.add(ids=ids, documents=docs, metadatas=metas)
                    total_chunks += len(ids)
                    ids, docs, metas = [], [], []
                    gc.collect()  # encourage Python to release memory

            # Final flush for this page
            if ids:
                col.add(ids=ids, documents=docs, metadatas=metas)
                total_chunks += len(ids)
                ids, docs, metas = [], [], []
                gc.collect()

        print(f"Indexed ~{total_chunks} chunks from {base}")
        gc.collect()

    print("Chroma build complete.")


if __name__ == "__main__":
    main()
