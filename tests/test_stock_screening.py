from fastapi.testclient import TestClient
import pandas as pd
from datetime import datetime, timedelta

from app.main import app
from app import main as main_mod


def test_screen_stocks_endpoint_with_monkeypatched_indicators(monkeypatch):
    """
    End-to-end via HTTP, but fully deterministic:
    - Monkeypatch market._download_map to return a tiny frame
    - Monkeypatch compute_indicators to force a bullish MACD crossover, low RSI, vol>VOL_SMA_20
    """

    # Access the single CoordinatorAgent instance used by the app
    agent = main_mod.agent
    market = agent.market

    # Build a tiny OHLCV frame (the actual values won't matter because indicators are patched)
    dates = pd.date_range(datetime.today() - timedelta(days=40), periods=30, freq="B")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000_000,
        }
    )

    def fake_download_map(_tickers):
        return {sym: df.copy() for sym in _tickers}

    # Patch the market data fetcher
    monkeypatch.setattr(market, "_download_map", fake_download_map)

    # Patch indicators to create an obvious pass case:
    # prev row: MACD <= SIGNAL ; last row: MACD > SIGNAL (bullish crossover)
    def fake_compute_indicators(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        n = len(out)
        out["RSI_14"] = [50.0] * (n - 1) + [35.0]  # last row below 40 → passes rsi_max=40
        out["MACD"] = [0.0] * (n - 1) + [1.0]
        out["MACD_SIGNAL"] = [0.1] * (n - 1) + [0.5]  # prev (0.1) <= 0.1; last 1.0 > 0.5
        out["SMA_50"] = [99.0] * n
        out["SMA_200"] = [98.0] * n
        out["VOL_SMA_20"] = [900_000] * n
        return out

    import app.utils.indicators as ind_mod
    monkeypatch.setattr(ind_mod, "compute_indicators", fake_compute_indicators)

    # Limit universe to deterministic small set for repeatability
    market.universe = ["AAPL", "MSFT", "GOOGL"]

    client = TestClient(app)
    payload = {
        "criteria": {
            "rsi_max": 40,
            "macd_signal": "bullish_crossover",
            "volume_increase": True
        }
    }
    r = client.post("/screen-stocks", json=payload)
    assert r.status_code == 200
    body = r.json()

    # Basic shape checks
    assert "bullish_stocks" in body and isinstance(body["bullish_stocks"], list)
    assert "total_screened" in body and body["total_screened"] == len(market.universe)
    assert len(body["bullish_stocks"]) > 0

    # The reasons should reflect our patched indicators
    reason_texts = " | ".join(p["reason"] for p in body["bullish_stocks"])
    assert "MACD bullish crossover" in reason_texts
    assert ("RSI oversold" in reason_texts) or ("RSI recovering" in reason_texts)
