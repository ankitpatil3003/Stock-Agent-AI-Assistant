# app/agents/knowledge.py
import os, re, logging
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from app.config.settings import Settings

logger = logging.getLogger(__name__)

# minimal typo fixes you’ve seen
_SPELL_FIX = {
    "Introduciton": "Introduction",
    "Fundamentalsofinvestments": "Fundamentals of Investments",
    "Iiiyearvisem": "III Year VI Sem",
}

# map common file stems → canonical names (add more as you ingest)
_ALIAS_BY_STEM = {
    "technical-analysis-and-stock-market-profits": "Technical Analysis and Stock Market Profits (Edwards & Magee)",
    "japanese-candlestick-charting": "Japanese Candlestick Charting Techniques (Nison)",
    "trading-for-a-living": "Trading for a Living (Elder)",
    "market-wizards": "Market Wizards (Schwager)",
    "a-complete-guide-to-volume-price-analysis": "Complete Guide to Volume Price Analysis (Coulling)",
}

def _apply_spellfix(s: str) -> str:
    for bad, good in _SPELL_FIX.items():
        s = re.sub(bad, good, s, flags=re.IGNORECASE)
    return s

def _sanitize(s: str | None) -> str:
    if not s:
        return ""
    s = re.sub(r"[\uFFFD\uFEFF]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _apply_spellfix(s)

def _friendly_title(meta: Dict[str, Any], doc: str) -> str:
    # prefer explicit metadata.title
    t = (meta or {}).get("title")
    if t:
        return _sanitize(t)

    # then file stem aliases
    src = (meta or {}).get("source") or (meta or {}).get("file") or ""
    stem = Path(src).stem if src else ""
    if stem:
        alias = _ALIAS_BY_STEM.get(stem.lower())
        if alias:
            return alias
        # turn “b.com(hons)...” into cleaner title-case
        t = stem.replace("_", " ").replace("-", " ")
        t = re.sub(r"\s+", " ", t).strip()
        t = _apply_spellfix(t)
        return t.title()

    # fallback: snippet
    snip = _sanitize((doc or "")[:80])
    return snip if snip else "Untitled Document"

class KnowledgeAgent:
    def __init__(self, settings: Settings):
        os.environ.setdefault("CHROMA_DISABLE_TELEMETRY", "1")
        self.settings = settings
        os.makedirs(self.settings.CHROMA_DIR, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.settings.CHROMA_DIR)
        self._collection = self._client.get_or_create_collection(
            name=self.settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = SentenceTransformer(self.settings.EMBEDDING_MODEL)
        logger.info(
            "KnowledgeAgent: using Chroma at '%s', collection='%s', model='%s'",
            self.settings.CHROMA_DIR,
            self.settings.CHROMA_COLLECTION,
            self.settings.EMBEDDING_MODEL,
        )

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        if not query:
            return []
        q_emb = self._embedder.encode([query], normalize_embeddings=True).tolist()[0]
        try:
            res = self._collection.query(
                query_embeddings=[q_emb],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Chroma query failed: {e}")
            return []

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        out: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            meta = meta or {}
            title = _friendly_title(meta, doc or "")
            score = (1.0 - float(dist)) if dist is not None else None
            out.append(
                {
                    "title": title,
                    "snippet": _sanitize((doc or "")[:240]),
                    "metadata": meta,
                    "score": score,
                }
            )
        return out
