import os
import glob
import uuid
import logging
from typing import List, Dict, Any, Tuple

from sentence_transformers import SentenceTransformer
from app.utils.vstore_faiss import FaissStore

# Prefer PyMuPDF; fallback to pdfminer if needed
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

logger = logging.getLogger(__name__)


def read_pdf(path: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    if PYMUPDF:
        with fitz.open(path) as doc:
            for i in range(len(doc)):
                text = doc.load_page(i).get_text("text")
                pages.append((i + 1, text or ""))
        return pages
    elif PDFMINER:
        text = pdfminer_extract_text(path) or ""
        chunks = text.split("\x0c")
        for i, t in enumerate(chunks, start=1):
            pages.append((i, t))
        return pages
    else:
        raise RuntimeError("No PDF parser available. Install PyMuPDF or pdfminer.six.")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        start = max(0, end - overlap)
        if start >= n:
            break
    return chunks


def build_faiss_from_pdfs(
    pdf_dir: str,
    out_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_name: str = "ta_pdfs",
) -> None:
    """
    One-shot embedding build:
      - Reads PDFs -> chunk -> embed (SentenceTransformer)
      - Builds FAISS IP index (cosine via normalization)
      - Saves index + docs + metas in out_dir
    """
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, f"{index_name}.faiss")
    docs_path = os.path.join(out_dir, f"{index_name}.docs.pkl")
    metas_path = os.path.join(out_dir, f"{index_name}.metas.pkl")

    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        logger.warning(f"No PDFs found in {pdf_dir}")
        return

    model = SentenceTransformer(model_name)

    all_chunks: List[str] = []
    all_metas: List[Dict[str, Any]] = []

    for pdf_path in pdf_paths:
        try:
            for page_no, text in read_pdf(pdf_path):
                for chunk in chunk_text(text):
                    all_chunks.append(chunk)
                    all_metas.append({"source": os.path.basename(pdf_path), "page": page_no})
        except Exception as e:
            logger.warning(f"Failed to read {pdf_path}: {e}")

    if not all_chunks:
        logger.warning("No text chunks extracted from PDFs.")
        return

    # Embed in batches
    embs = model.encode(all_chunks, batch_size=64, convert_to_numpy=True, show_progress_bar=True)

    store = FaissStore(index_path=index_path, docs_path=docs_path, metas_path=metas_path)
    store.build(embeddings=embs, docs=all_chunks, metas=all_metas)
    store.save()

    logger.info(f"FAISS index built with {len(all_chunks)} chunks → {index_path}")
