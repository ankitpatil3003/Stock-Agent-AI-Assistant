# app/agents/coordinator.py
from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.agents.knowledge import KnowledgeAgent
from app.agents.market_data import MarketDataAgent
from app.config.settings import Settings

# Optional OpenAI (safe if key not set)
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

_TA_KEYWORDS = [
    "rsi", "macd", "moving average", "sma", "ema", "divergence",
    "candlestick", "support", "resistance", "breakout", "trend",
]
_EXCLUDE_HINTS = [
    "lecture", "week", "semester", "b.com", "assignment", "exam", "notes",
    "introduction to", "fundamentals of investments",
    "ethos", "thesis",
]

def _is_excluded_title(t: str) -> bool:
    t_low = t.lower()
    return any(h in t_low for h in _EXCLUDE_HINTS)

def _keyword_score(t: str) -> int:
    t_low = t.lower()
    return sum(1 for k in _TA_KEYWORDS if k in t_low)


class CoordinatorAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.knowledge = KnowledgeAgent(settings=settings)
        self.market = MarketDataAgent(settings=settings)

        self._openai_client = None
        if _OPENAI_AVAILABLE and getattr(self.settings, "OPENAI_API_KEY", None):
            try:
                self._openai_client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
                logger.info("OpenAI client initialized.")
            except Exception as e:  # pragma: no cover
                logger.warning(f"OpenAI initialization failed: {e}")

    # -------- Analyze Strategy --------
    def analyze_strategy(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        strategy = (payload.get("strategy") or "").strip()
        win_rate = payload.get("win_rate")
        avg_return = payload.get("avg_return")
        sample_stocks: List[str] = (payload.get("sample_stocks") or [])[:3]

        # Knowledge hits
        hits = self.knowledge.search(strategy, k=12)
        hits = [h for h in hits if (h.get("score") or 0) >= 0.35]

        # Score + filter titles
        scored: List[Dict[str, Any]] = []
        for h in hits:
            t = (h.get("title") or "").strip()
            if not t or _is_excluded_title(t):
                continue
            kw = _keyword_score(t)
            total = float(h.get("score") or 0) + 0.05 * kw
            scored.append({
                "title": t,
                "score": total,
                "metadata": h.get("metadata", {}),
                "snippet": h.get("snippet", ""),
            })

        # Sort + dedupe
        seen, titles = set(), []
        for item in sorted(scored, key=lambda x: x["score"], reverse=True):
            key = item["title"].casefold()
            if key not in seen:
                titles.append(item["title"])
                seen.add(key)
            if len(titles) >= 5:
                break

        if not titles:
            titles = ["RSI Divergence", "RSI with Moving Average", "RSI + Volume Confirmation"]

        # Market context
        market_ctx = self.market.get_recent_context(sample_stocks)

        # Improvements (LLM if available, else heuristics)
        improvements = self._generate_improvements(payload, scored, market_ctx)

        return {
            "improvements": improvements[:5],
            "similar_strategies": titles,
            "recent_performance": market_ctx,
            "meta": {"win_rate": win_rate, "avg_return": avg_return},
        }

    # -------- Stock Screening --------
    def screen_stocks(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        picks = self.market.screen(criteria)
        picks = sorted(picks, key=lambda x: x.get("score", 0), reverse=True)[:10]

        total = len(getattr(self.market, "universe", []))
        max_tickers = getattr(self.settings, "MAX_TICKERS", total or 0)

        return {
            "bullish_stocks": picks,
            "total_screened": min(total, max_tickers) if total and max_tickers else total,
            "timestamp": self.market.now_iso() if hasattr(self.market, "now_iso") else None,
        }

    # -------- Improvements helpers --------
    def _generate_improvements(
        self,
        payload: Dict[str, Any],
        similar_hits: List[Dict[str, Any]],
        market_ctx: Dict[str, Any],
    ) -> List[str]:
        # Use OpenAI if configured
        if self._openai_client and getattr(self.settings, "OPENAI_MODEL", None) and getattr(self.settings, "OPENAI_API_KEY", None):
            try:
                prompt = self._build_llm_prompt(payload, similar_hits, market_ctx)
                resp = self._openai_client.chat.completions.create(
                    model=self.settings.OPENAI_MODEL,
                    temperature=0.3,
                    messages=[
                        {"role": "system",
                         "content": "You are a pragmatic trading assistant. Offer concrete, risk-aware, low-cost suggestions. Keep to 3–5 bullets."},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = (resp.choices[0].message.content or "").strip()
                bullets = []
                for line in text.splitlines():
                    clean = line.strip(" -•\t")
                    if clean:
                        bullets.append(clean)
                if bullets:
                    return bullets[:5]
            except Exception as e:  # pragma: no cover
                logger.warning(f"LLM generation failed, falling back to heuristics: {e}")

        # Fallback to heuristics
        return self._heuristic_improvements(payload)

    def _build_llm_prompt(
        self,
        payload: Dict[str, Any],
        similar_hits: List[Dict[str, Any]],
        market_ctx: Dict[str, Any],
    ) -> str:
        hits_lines = []
        for h in similar_hits[:5]:
            meta = h.get("metadata") or {}
            src = meta.get("source") or meta.get("file") or "?"
            page = meta.get("page", "?")
            snip = (h.get("snippet") or "")[:180]
            hits_lines.append(f"- {src} p{page}: {snip}")

        market_lines = []
        for sym, ctx in market_ctx.items():
            rsi = ctx.get("rsi")
            rsi_str = f"{rsi:.1f}" if isinstance(rsi, (int, float)) else "?"
            market_lines.append(f"- {sym}: RSI={rsi_str}, trend={ctx.get('trend','?')}")

        return (
            f"Strategy: {payload.get('strategy','')}\n"
            f"Win rate: {payload.get('win_rate')}, Avg return: {payload.get('avg_return')}\n"
            f"Similar passages:\n" + ("\n".join(hits_lines) if hits_lines else "(none)") + "\n"
            f"Market context:\n" + ("\n".join(market_lines) if market_lines else "(none)") + "\n\n"
            "Provide 3–5 specific improvements (risk controls, filters, exits, validation)."
        )

    def _heuristic_improvements(self, payload: Dict[str, Any]) -> List[str]:
        s = (payload.get("strategy") or "").lower()
        tips: List[str] = []

        if "rsi" in s:
            tips += [
                "Use 14-period RSI with a higher-timeframe trend filter (e.g., trade long only when price > 200-SMA).",
                "Confirm RSI < 30 entries with MACD cross or volume expansion to reduce false starts.",
            ]
        if "macd" in s:
            tips += [
                "Require MACD histogram to tick up for 2 consecutive bars to avoid whipsaws.",
                "Avoid entries right into support/resistance; wait for break and retest.",
            ]
        if any(k in s for k in ["sma", "ema", "moving average"]):
            tips += [
                "Add a volatility gate: only trade when ATR(14)/Close > 1.5%.",
                "Use a time stop (exit after N bars if target not hit).",
            ]

        tips += [
            "Risk ≤1% per trade using ATR(14)-based stops.",
            "Walk-forward validate on the last 12 months to ensure robustness.",
            "Size positions inversely to volatility (smaller size for higher ATR).",
        ]

        # de-dupe while preserving order
        seen, uniq = set(), []
        for t in tips:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        return uniq[:5]
