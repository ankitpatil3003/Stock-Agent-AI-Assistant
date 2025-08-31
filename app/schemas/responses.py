from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class AnalyzeStrategyResponse(BaseModel):
    improvements: List[str]
    similar_strategies: List[str]
    recent_performance: Dict[str, Dict[str, Optional[float] | str]]


class StockPick(BaseModel):
    symbol: str
    score: float
    rsi: Optional[float] = None
    reason: str


class ScreenStocksResponse(BaseModel):
    bullish_stocks: List[StockPick]
    total_screened: int
    timestamp: str
