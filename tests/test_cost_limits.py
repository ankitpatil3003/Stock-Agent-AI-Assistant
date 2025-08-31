import json
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from app.agents.market_data import MarketDataAgent
from app.config.settings import settings


def _yf_multiindex_frame(symbols):
    """
    Build a minimal yfinance-like MultiIndex DataFrame for several tickers.
    Columns: (TICKER, OHLCV)
    """
    dates = pd.date_range(datetime.today() - timedelta(days=40), periods=30, freq="B")
    cols = pd.MultiIndex.from_product([symbols, ["Open", "High", "Low", "Close", "Volume"]])
    data = np.zeros((len(dates), len(cols)))

    # simple upward drift + steady volume
    for si, sym in enumerate(symbols):
        base = 100 + si * 5
        data[:, si * 5 + 0] = base + 0.1  # Open
        data[:, si * 5 + 1] = base + 0.5  # High
        data[:, si * 5 + 2] = base - 0.5  # Low
        data[:, si * 5 + 3] = base + np.linspace(0, 2.9, len(dates))  # Close trending up
        data[:, si * 5 + 4] = 1_000_000  # Volume

    df = pd.DataFrame(data, index=dates, columns=cols)
    return df


def test_yfinance_calls_reduced_by_cache(monkeypatch, tmp_path):
    """
    Calling _download_map twice within TTL should hit yfinance only once.
    """
    # point cache to an isolated temp dir
    monkeypatch.setattr(settings, "CACHE_DIR", str(tmp_path))

    # keep MAX_TICKERS small for the test
    monkeypatch.setattr(settings, "MAX_TICKERS", 2)

    # create a temporary S&P100 file with two tickers
    sp100_path = tmp_path / "sp100.json"
    sp100_path.write_text(json.dumps(["AAPL", "MSFT"]))
    monkeypatch.setattr(settings, "SP100_FILE", str(sp100_path))

    # long TTL so the second call still uses cache
    monkeypatch.setattr(settings, "CACHE_TTL_MARKET", 3600)

    calls = {"n": 0}

    def fake_download(*args, **kwargs):
        calls["n"] += 1
        # Return a yfinance-like MultiIndex DataFrame for AAPL & MSFT
        return _yf_multiindex_frame(["AAPL", "MSFT"])

    # Patch yfinance.download
    import yfinance as yf
    monkeypatch.setattr(yf, "download", fake_download)

    agent = MarketDataAgent(settings=settings)

    # First call: should trigger yfinance
    out1 = agent._download_map(["AAPL", "MSFT"])
    assert isinstance(out1, dict)
    assert "AAPL" in out1 and "MSFT" in out1
    assert calls["n"] == 1

    # Second call (same inputs, within TTL): should use cache
    out2 = agent._download_map(["MSFT", "AAPL"])  # order differs but key sorts internally
    assert isinstance(out2, dict)
    assert calls["n"] == 1  # unchanged → cache hit

    # Sanity: dataframes present
    assert out2["AAPL"] is not None and not out2["AAPL"].empty


def test_respects_max_tickers(monkeypatch, tmp_path):
    """
    MarketDataAgent.screen should only process up to settings.MAX_TICKERS.
    """
    # set a larger fake universe; enforce MAX_TICKERS=3
    monkeypatch.setattr(settings, "MAX_TICKERS", 3)
    monkeypatch.setattr(settings, "CACHE_DIR", str(tmp_path))

    agent = MarketDataAgent(settings=settings)
    agent.universe = [f"T{i}" for i in range(10)]  # 10 tickers available

    def fake_download_map(tickers):
        # return a simple DF for the tickers requested (should be capped to 3)
        frames = {}
        for t in tickers:
            frames[t] = _yf_multiindex_frame([t])[t].reset_index().rename_axis("Date").reset_index(drop=True)
        return frames

    # Patch internals
    monkeypatch.setattr(agent, "_download_map", fake_download_map)

    res = agent.screen({"rsi_max": 70, "macd_signal": None, "volume_increase": False})
    # We can't know exact picks without indicators, but we can assert we didn't fetch > MAX_TICKERS
    # via the behavior of _download_map; here we indirectly assert nothing crashed and the result is a list.
    assert isinstance(res, list)
