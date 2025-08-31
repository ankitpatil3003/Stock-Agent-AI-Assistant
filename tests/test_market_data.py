import pandas as pd
from datetime import datetime, timedelta

from app.agents.market_data import MarketDataAgent
from app.config.settings import settings

def _fake_df():
    dates = pd.date_range(datetime.today() - timedelta(days=120), periods=90, freq="B")
    return pd.DataFrame({
        "Date": dates,
        "Open": 100.0,
        "High": 101.0,
        "Low":  99.0,
        "Close": 100.0 + (pd.Series(range(len(dates))) * 0.1),
        "Volume": 1_000_000
    })

def test_screen_monkeypatched(monkeypatch):
    agent = MarketDataAgent(settings=settings)

    def fake_download_map(_tickers):
        return {sym: _fake_df() for sym in _tickers[:5]}

    monkeypatch.setattr(agent, "_download_map", fake_download_map)

    res = agent.screen({"rsi_max": 60, "macd_signal": "bullish_crossover", "volume_increase": False})
    assert isinstance(res, list)
    # no crash + score present
    if res:
        assert "symbol" in res[0] and "score" in res[0]
