import types
from fastapi.testclient import TestClient
from app.main import app
from app.agents.coordinator import CoordinatorAgent

def test_analyze_strategy_endpoint(monkeypatch):
    client = TestClient(app)

    # Monkeypatch coordinator methods to avoid external deps
    from app import main as main_mod
    agent: CoordinatorAgent = main_mod.agent

    async def fake_analyze(req):
        return types.SimpleNamespace(
            improvements=["Tip 1", "Tip 2"],
            similar_strategies=["doc p1: ..."],
            recent_performance={"AAPL": {"rsi": 45.0, "trend": "neutral"}}
        )

    monkeypatch.setattr(agent, "analyze_strategy", fake_analyze)

    payload = {
        "strategy": "RSI crossover",
        "win_rate": 0.6,
        "avg_return": 0.05,
        "sample_stocks": ["AAPL"]
    }
    r = client.post("/analyze-strategy", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "improvements" in body and "similar_strategies" in body and "recent_performance" in body
