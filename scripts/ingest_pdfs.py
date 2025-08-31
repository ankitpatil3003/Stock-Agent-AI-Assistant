# app/agents/coordinator.py
from typing import Any, Dict, List
import re

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
    ...
    def analyze_strategy(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        strategy = payload.get("strategy", "")
        win_rate = payload.get("win_rate")
        avg_return = payload.get("avg_return")
        sample_stocks: List[str] = (payload.get("sample_stocks") or [])[:3]

        hits = self.knowledge.search(strategy, k=12)

        # keep only reasonably similar items
        hits = [h for h in hits if (h.get("score") or 0) >= 0.35]

        # build scored list, exclude lecture/intro items
        scored: List[Dict[str, Any]] = []
        for h in hits:
            t = (h.get("title") or "").strip()
            if not t or _is_excluded_title(t):
                continue
            kw = _keyword_score(t)
            # combine cosine score with keyword bonus
            total = float(h.get("score") or 0) + 0.05 * kw
            scored.append({"title": t, "score": total})

        # sort by total score desc and dedupe by title
        seen, titles = set(), []
        for item in sorted(scored, key=lambda x: x["score"], reverse=True):
            k = item["title"].casefold()
            if k not in seen:
                titles.append(item["title"])
                seen.add(k)
            if len(titles) >= 5:
                break

        if not titles:
            titles = ["RSI Divergence", "RSI with Moving Average", "RSI + Volume Confirmation"]

        market_ctx = self.market.describe_symbols(sample_stocks)

        improvements = [
            "Implement a MACD-below-zero requirement when buying RSI < 30 to avoid counter-trend entries.",
            "Use an ATR-based stop-loss and a trailing take-profit to systematize risk and exits.",
            "Validate on multiple market regimes (bear/bull/sideways) to detect regime sensitivity.",
            "Filter longs to when price > 200-SMA (and shorts when < 200-SMA).",
            "Require above-average volume on entry to confirm momentum.",
        ][:5]

        return {
            "improvements": improvements,
            "similar_strategies": titles,
            "recent_performance": market_ctx,
            "meta": {"win_rate": win_rate, "avg_return": avg_return},
        }
