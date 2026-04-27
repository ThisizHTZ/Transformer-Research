import yfinance as yf
import pandas as pd


def download_prices(tickers, start_date, end_date=None):
    frames = []

    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            print(f"[WARN] No data for {ticker}")
            continue

        df = df.reset_index()
        df["ticker"] = ticker

        # yfinance sometimes returns multi-level columns.
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "ticker"]
        df = df[keep_cols]
        df = df.rename(columns={"Date": "date"})
        frames.append(df)

    if not frames:
        raise ValueError("No data downloaded. Check tickers or internet connection.")

    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out
