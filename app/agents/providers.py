# app/agents/providers.py
import os
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/api/v3"
FINNHUB_BASE = "https://finnhub.io/api/v1"


def _build_session() -> requests.Session:
    """Robust HTTP session with retries and a friendly UA."""
    sess = requests.Session()
    retries = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.headers.update({
        "User-Agent": os.getenv(
            "USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124 Safari/537.36"
        )
    })
    return sess


_SESS = _build_session()


def _req_json(url: str, params: Dict[str, str], timeout: int = 12) -> Optional[dict]:
    try:
        r = _SESS.get(url, params=params, timeout=timeout)
        if r.status_code == 429:
            # tiny backoff then one retry
            time.sleep(1.0)
            r = _SESS.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        body = None
        try:
            body = r.text[:300] if "r" in locals() and hasattr(r, "text") else None
        except Exception:
            pass
        log.warning(f"HTTP error for {url} params={params} err={e} body={body!r}")
        return None


def fmp_daily_df(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """
    Daily EOD candles (~1y) from FMP.
    IMPORTANT: do NOT use serietype=line (it strips O/H/L/V). We need full OHLCV.
    Endpoint: /historical-price-full/{symbol}?timeseries=N
    """
    key = os.getenv("FMP_API_KEY", "")
    if not key:
        log.error("FMP_API_KEY missing")
        return None

    url = f"{FMP_BASE}/historical-price-full/{symbol.upper()}"
    # allow a little buffer
    n = max(30, min(days + 10, 370))
    params = {"timeseries": str(n), "apikey": key}
    js = _req_json(url, params)
    if not js or "historical" not in js:
        log.warning(f"FMP: no 'historical' for {symbol}")
        return None

    rows = js.get("historical") or []
    if not rows:
        log.warning(f"FMP: empty historical for {symbol}")
        return None

    df = pd.DataFrame(rows)
    # Expect: date, open, high, low, close, volume
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        log.warning(f"FMP: missing columns for {symbol}; got {set(df.columns)}")
        return None

    # Normalize
    df["Date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.dropna().sort_values("Date").reset_index(drop=True)

    # Filter to requested window if needed (optional; timeseries already caps)
    if days and len(df) > 0:
        start_cut = df["Date"].max() - timedelta(days=days + 3)
        df = df[df["Date"] >= start_cut].reset_index(drop=True)

    if df.empty:
        log.warning(f"FMP: resulting DF empty for {symbol}")
        return None

    return df


def finnhub_quote(symbol: str) -> Optional[Dict[str, float]]:
    """
    Latest quote from Finnhub; returns dict with last price and day OHLC if available.
    """
    key = os.getenv("FINNHUB_API_KEY", "")
    if not key:
        return None
    url = f"{FINNHUB_BASE}/quote"
    js = _req_json(url, {"symbol": symbol.upper(), "token": key})
    if not js or "c" not in js:
        log.warning(f"Finnhub: no quote for {symbol}")
        return None
    # c= current, h= high, l= low, o= open, t= epoch
    try:
        return {
            "last": float(js.get("c") or 0.0),
            "open": float(js.get("o") or 0.0),
            "high": float(js.get("h") or 0.0),
            "low":  float(js.get("l") or 0.0),
            "ts":   int(js.get("t") or 0),
        }
    except Exception as e:
        log.warning(f"Finnhub: parse error for {symbol}: {e} payload={js}")
        return None
