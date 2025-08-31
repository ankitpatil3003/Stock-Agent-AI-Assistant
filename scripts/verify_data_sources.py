"""
Quick smoke checks:
- PDFs presence
- Chroma directory writable
- Market universe file
- Can compute indicators on a tiny synthetic dataframe
"""
import os
import json
import pandas as pd
from datetime import datetime, timedelta

from app.config.settings import settings, init_logging
from app.utils.indicators import compute_indicators

def main():
    init_logging()

    # PDFs
    pdfs_ok = os.path.isdir(settings.PDF_DIR) and len([p for p in os.listdir(settings.PDF_DIR) if p.lower().endswith(".pdf")]) > 0
    print(f"PDFs present: {pdfs_ok} ({settings.PDF_DIR})")

    # Chroma path
    os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
    print(f"Chroma path OK: {settings.VECTOR_DB_PATH}")

    # Universe
    if os.path.exists(settings.SP100_FILE):
        with open(settings.SP100_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"S&P100 file OK: {len(data)} tickers")
    else:
        print("S&P100 file missing; agent will use fallback mini universe.")

    # Indicators quick check on synthetic series
    dates = pd.date_range(datetime.today() - timedelta(days=260), periods=220, freq="B")
    df = pd.DataFrame({
        "Date": dates,
        "Open": 100.0,
        "High": 101.0,
        "Low":  99.0,
        "Close": 100.0 + (pd.Series(range(len(dates))) * 0.05),
        "Volume": 1_000_000,
    })
    out = compute_indicators(df)
    print(f"Indicators computed: columns -> {list(out.columns)}; rows={len(out)}")

if __name__ == "__main__":
    main()
