# app/schemas/requests.py
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class AnalyzeStrategyRequest(BaseModel):
    strategy: str = Field(..., description="Natural-language strategy rules")
    win_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    avg_return: Optional[float] = Field(
        None, description="Average per-trade return (decimal, e.g., 0.08 = 8%)"
    )
    # key change: default_factory=list so it's [] instead of None
    sample_stocks: List[str] = Field(
        default_factory=list, description="Up to 3 tickers for quick market context"
    )

class ScreenCriteria(BaseModel):
    rsi_max: Optional[float] = Field(None, description="Max RSI to consider (e.g., 40)")
    macd_signal: Optional[Literal["bullish_crossover", "bearish_crossover"]] = None
    volume_increase: Optional[bool] = Field(
        default=False, description="Require volume > SMA(20)"
    )

class ScreenStocksRequest(BaseModel):
    criteria: ScreenCriteria
