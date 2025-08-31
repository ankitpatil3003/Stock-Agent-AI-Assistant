import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import pandas as pd

from app.agents.providers import fmp_daily_df, finnhub_quote
from app.utils.indicators import compute_indicators
from app.utils.cache import FileCache
from app.config.settings import Settings

logger = logging.getLogger(__name__)


class MarketDataAgent:
    """
    Daily OHLCV + indicators for S&P100 universe.
    Provider chain (no yfinance):
      1) FMP (Financial Modeling Prep) for 1y daily history (EOD)
      2) Finnhub (optional) to refresh the latest close with a near-real-time last price
    Caches only on success.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.universe: List[str] = self._load_universe(self.settings.SP100_FILE)
        self.cache = FileCache(os.path.join(self.settings.CACHE_DIR, "market"))

        # Make sure downstream helpers that rely on env vars can see the keys
        if self.settings.FMP_API_KEY and not os.getenv("FMP_API_KEY"):
            os.environ["FMP_API_KEY"] = self.settings.FMP_API_KEY
        if self.settings.FINNHUB_API_KEY and not os.getenv("FINNHUB_API_KEY"):
            os.environ["FINNHUB_API_KEY"] = self.settings.FINNHUB_API_KEY

        # Use settings (not os.getenv) for warnings/info
        if not self.settings.FMP_API_KEY:
            logger.warning("FMP_API_KEY not set; market data will be unavailable.")
        if not self.settings.FINNHUB_API_KEY:
            logger.info("FINNHUB_API_KEY not set; results will use EOD close only (no live quote refresh).")

        logger.info(f"Universe size = {len(self.universe)} tickers")

    # ---------- public ----------

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def get_recent_context(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Returns a symbol -> {rsi, macd, macd_signal, sma50, sma200, close, trend} map
        computed from the latest row of indicators.
        """
        ctx: Dict[str, Any] = {}
        if not tickers:
            return ctx

        df_map = self._download_map(tickers)
        for sym, df in df_map.items():
            if df is None or df.empty:
                continue

            ind = compute_indicators(df)
            last = ind.iloc[-1]

            rsi = float(last["RSI_14"]) if "RSI_14" in ind.columns and pd.notna(last["RSI_14"]) else None
            macd = float(last["MACD"]) if "MACD" in ind.columns and pd.notna(last["MACD"]) else None
            macd_sig = float(last["MACD_SIGNAL"]) if "MACD_SIGNAL" in ind.columns and pd.notna(last["MACD_SIGNAL"]) else None
            sma50 = float(last["SMA_50"]) if "SMA_50" in ind.columns and pd.notna(last["SMA_50"]) else None
            sma200 = float(last["SMA_200"]) if "SMA_200" in ind.columns and pd.notna(last["SMA_200"]) else None
            close = float(last["Close"]) if "Close" in ind.columns and pd.notna(last["Close"]) else None

            trend = "neutral"
            if rsi is not None and rsi >= 70:
                trend = "overbought"
            elif rsi is not None and rsi <= 30:
                trend = "oversold"
            elif close is not None and sma50 is not None and sma200 is not None:
                if close > sma50 > sma200:
                    trend = "bullish"
                elif close < sma50 < sma200:
                    trend = "bearish"

            ctx[sym] = {
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_sig,
                "sma50": sma50,
                "sma200": sma200,
                "close": close,
                "trend": trend,
            }
        return ctx

    def screen(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Screens the universe with simple technical criteria.
        Input can be a dict or a Pydantic model; we normalize to dict here.
        """
        if not isinstance(criteria, dict):
            if hasattr(criteria, "model_dump"):
                criteria = criteria.model_dump()
            elif hasattr(criteria, "dict"):
                criteria = criteria.dict()
            else:
                try:
                    criteria = dict(criteria)
                except Exception:
                    criteria = {}

        rsi_max = criteria.get("rsi_max")
        macd_sig = criteria.get("macd_signal")
        vol_inc = criteria.get("volume_increase", False)

        tickers = self.universe[: self.settings.MAX_TICKERS]
        df_map = self._download_map(tickers)

        picks: List[Dict[str, Any]] = []
        for sym, df in df_map.items():
            if df is None or df.empty or len(df) < 35:
                continue

            ind = compute_indicators(df)
            last = ind.iloc[-1]
            prev = ind.iloc[-2] if len(ind) >= 2 else None

            rsi = float(last["RSI_14"]) if "RSI_14" in ind.columns and pd.notna(last["RSI_14"]) else None
            macd = float(last["MACD"]) if "MACD" in ind.columns and pd.notna(last["MACD"]) else None
            macd_signal = float(last["MACD_SIGNAL"]) if "MACD_SIGNAL" in ind.columns and pd.notna(last["MACD_SIGNAL"]) else None
            vol = float(last["Volume"]) if "Volume" in ind.columns and pd.notna(last["Volume"]) else None
            vol_sma20 = float(last["VOL_SMA_20"]) if "VOL_SMA_20" in ind.columns and pd.notna(last["VOL_SMA_20"]) else None

            # RSI threshold
            if rsi is not None and rsi_max is not None and rsi > float(rsi_max):
                continue

            # MACD crossovers
            cross_up = False
            cross_down = False
            if prev is not None:
                prev_macd = float(prev["MACD"]) if "MACD" in ind.columns and pd.notna(prev["MACD"]) else None
                prev_sig = float(prev["MACD_SIGNAL"]) if "MACD_SIGNAL" in ind.columns and pd.notna(prev["MACD_SIGNAL"]) else None
                if prev_macd is not None and prev_sig is not None and macd is not None and macd_signal is not None:
                    cross_up = prev_macd <= prev_sig and macd > macd_signal
                    cross_down = prev_macd >= prev_sig and macd < macd_signal

            if macd_sig == "bullish_crossover" and not cross_up:
                continue
            if macd_sig == "bearish_crossover" and not cross_down:
                continue

            # Volume expansion
            if vol_inc and (vol is None or vol_sma20 is None or vol <= vol_sma20):
                continue

            # Simple scoring
            score = 0.0
            if rsi is not None:
                score += max(0.0, (60 - rsi) / 30.0) * 3.0
            if cross_up:
                score += 3.0
            if vol_inc and vol is not None and vol_sma20 is not None:
                score += min(3.0, (vol / vol_sma20) - 1.0)
            last_close = float(last["Close"]) if "Close" in ind.columns and pd.notna(last["Close"]) else None
            sma50 = float(last["SMA_50"]) if "SMA_50" in ind.columns and pd.notna(last["SMA_50"]) else None
            if sma50 is not None and last_close is not None and last_close > sma50:
                score += 1.0

            picks.append(
                {
                    "symbol": sym,
                    "score": round(float(score), 2),
                    "rsi": round(float(rsi), 1) if rsi is not None else None,
                    "reason": self._reason_text(rsi, cross_up, vol, vol_sma20),
                }
            )

        return picks

    # ---------- internals ----------

    def _load_universe(self, file_path: str) -> List[str]:
        try:
            if file_path and os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                syms = [s.strip().upper() for s in data if isinstance(s, str)]
                return sorted(list(set(syms)))
        except Exception as e:
            logger.warning(f"Failed to load S&P100 file: {e}")

        fallback = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META",
            "NVDA", "TSLA", "BRK-B", "JPM", "XOM", "UNH", "JNJ",
        ]
        logger.warning("Using fallback mini universe (set SP100_FILE to override).")
        return fallback

    def _reason_text(self, rsi, cross_up, vol, vol_sma20) -> str:
        parts = []
        if rsi is not None:
            if rsi < 35:
                parts.append("RSI oversold")
            elif rsi < 45:
                parts.append("RSI recovering")
        if cross_up:
            parts.append("MACD bullish crossover")
        if vol is not None and vol_sma20 is not None and vol > vol_sma20:
            parts.append("volume expansion")
        return " + ".join(parts) if parts else "meets criteria"

    def _download_map(self, tickers: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Strategy: cache -> FMP per-symbol (EOD 1y) -> (optional) Finnhub enrich last close.
        Cache only if at least one symbol succeeded.
        """
        if not tickers:
            return {}

        days = int(365 * max(1, self.settings.HISTORY_YEARS))
        key = f"prices:FMP:{days}:{','.join(sorted(set(tickers)))}"
        cached = self.cache.get(key, ttl_sec=self.settings.CACHE_TTL_MARKET)
        if cached is not None:
            return cached

        result: Dict[str, Optional[pd.DataFrame]] = {}

        # 1) FMP per-symbol history
        for sym in tickers:
            df = fmp_daily_df(sym, days=days)
            result[sym] = df

        # 2) Finnhub quote to “freshen” the last Close (if available)
        have_any = False
        use_finnhub = bool(os.getenv("FINNHUB_API_KEY"))
        for sym, df in result.items():
            if df is None or df.empty:
                continue
            have_any = True
            if use_finnhub:
                q = finnhub_quote(sym)
                if q and q.get("last", 0.0) > 0.0:
                    last_idx = df.index[-1]
                    df.at[last_idx, "Close"] = float(q["last"])
                    result[sym] = df

        if have_any:
            self.cache.set(key, result)
        else:
            logger.error("FMP returned no data for all requested symbols; not caching.")
        return result
