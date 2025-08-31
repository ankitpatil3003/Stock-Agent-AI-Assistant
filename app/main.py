import os
import inspect
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import pandas as pd

from app.schemas.requests import AnalyzeStrategyRequest, ScreenStocksRequest
from app.schemas.responses import (
    AnalyzeStrategyResponse,
    ScreenStocksResponse,
    HealthResponse,
)
from app.config.settings import settings, init_logging
from app.agents.coordinator import CoordinatorAgent
from app.utils.indicators import compute_indicators

os.environ.setdefault("CHROMA_DISABLE_TELEMETRY", "1")

# Init logging first
init_logging()

app = FastAPI(
    title="Technical Analysis RAG Agent",
    version="0.1.0",
    description="Basic RAG + yfinance-based strategy analyzer and stock screener.",
)

agent = CoordinatorAgent(settings=settings)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/analyze-strategy", response_model=AnalyzeStrategyResponse)
async def analyze_strategy(req: AnalyzeStrategyRequest) -> AnalyzeStrategyResponse:
    try:
        payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        payload["sample_stocks"] = (payload.get("sample_stocks") or [])[:3]

        # Call agent and await if needed
        res = agent.analyze_strategy(payload)
        if inspect.isawaitable(res):
            res = await res

        # Normalize to the declared response model
        if isinstance(res, AnalyzeStrategyResponse):
            return res
        if isinstance(res, dict):
            return AnalyzeStrategyResponse(**res)
        if hasattr(res, "__dict__"):  # SimpleNamespace, dataclass-like
            return AnalyzeStrategyResponse(**res.__dict__)

        # As a last resort, let Pydantic coerce if it's already close to a dict
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/screen-stocks", response_model=ScreenStocksResponse)
def screen_stocks(req: ScreenStocksRequest) -> ScreenStocksResponse:
    try:
        payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        # Pass only the criteria dict to the agent
        criteria = payload.get("criteria") or {}
        return agent.screen_stocks(criteria)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/market")
def debug_market(ticker: str = Query(..., min_length=1, max_length=10)):
    sym = ticker.upper().strip()
    df_map = agent.market._download_map([sym])
    df = df_map.get(sym)

    if df is None or df.empty:
        return JSONResponse({"ticker": sym, "status": "no_data"}, status_code=200)

    ind = compute_indicators(df)
    if ind is None or ind.empty:
        return JSONResponse({"ticker": sym, "status": "no_indicators"}, status_code=200)

    last = ind.iloc[-1]
    payload = {
        "ticker": sym,
        "status": "ok",
        "close": float(last.get("Close")) if pd.notna(last.get("Close")) else None,
        "rsi": float(last.get("RSI_14")) if pd.notna(last.get("RSI_14")) else None,
        "macd": float(last.get("MACD")) if pd.notna(last.get("MACD")) else None,
        "macd_signal": float(last.get("MACD_SIGNAL")) if pd.notna(last.get("MACD_SIGNAL")) else None,
        "sma50": float(last.get("SMA_50")) if pd.notna(last.get("SMA_50")) else None,
        "sma200": float(last.get("SMA_200")) if pd.notna(last.get("SMA_200")) else None,
        "vol_sma20": float(last.get("VOL_SMA_20")) if pd.notna(last.get("VOL_SMA_20")) else None,
    }
    return JSONResponse(payload, status_code=200)


# --- DEBUG: provider + keys presence ---
@app.get("/debug/provider")
def debug_provider():
    prov = os.getenv("MARKET_PROVIDER", "auto")
    fmp = bool(os.getenv("FMP_API_KEY"))
    fhub = bool(os.getenv("FINNHUB_API_KEY"))
    return {
        "market_provider_env": prov,
        "has_FMP_API_KEY": fmp,
        "has_FINNHUB_API_KEY": fhub,
        "cache_dir": str(agent.market.cache.cache_dir) if hasattr(agent.market, "cache") else None,
        "universe_size": len(agent.market.universe),
    }
