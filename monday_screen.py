
from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf
import math

def get_constituents():
    frames = []
    for index_name, url in [
        ("FTSE 100", "https://en.wikipedia.org/wiki/FTSE_100_Index"),
        ("FTSE 250", "https://en.wikipedia.org/wiki/FTSE_250_Index"),
    ]:
        for t in pd.read_html(url):
            cols = [str(c).lower() for c in t.columns]
            if any("ticker" in c or "epic" in c for c in cols) and any("company" in c for c in cols):
                ticker_col = next(c for c in t.columns if "ticker" in str(c).lower() or "epic" in str(c).lower())
                company_col = next(c for c in t.columns if "company" in str(c).lower())
                out = t[[company_col, ticker_col]].copy()
                out.columns = ["Company", "Ticker"]
                out["Index"] = index_name
                out["Ticker"] = out["Ticker"].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip().str.replace(".", "-", regex=False) + ".L"
                frames.append(out)
                break
    return pd.concat(frames, ignore_index=True).drop_duplicates("Ticker")

def eps_series(qf):
    for row in ["Diluted EPS", "Basic EPS", "Normalized EPS"]:
        if row in qf.index:
            s = pd.to_numeric(qf.loc[row], errors="coerce").dropna()
            if len(s) >= 4:
                return s
    return None

def latest4(s):
    x = sorted([(pd.to_datetime(k), float(v)) for k, v in s.items()], reverse=True)
    return x[:4]

def qualifies(v):
    q0, q1, q2, q3 = v
    return q0 > q1 > q2 and not (q2 > q3)

def main():
    rows = []
    for _, r in get_constituents().iterrows():
        try:
            t = yf.Ticker(r["Ticker"])
            info = t.info or {}
            pe = info.get("trailingPE")
            if pe is None:
                p = info.get("currentPrice") or info.get("regularMarketPrice")
                e = info.get("trailingEps")
                if p and e and e > 0:
                    pe = p / e
            if pe is None or not isinstance(pe, (int, float)) or not math.isfinite(pe) or pe <= 0:
                continue

            s = eps_series(t.quarterly_financials)
            if s is None:
                continue
            four = latest4(s)
            vals = [v for _, v in four]
            if qualifies(vals):
                rows.append({
                    "Company": r["Company"],
                    "Ticker": r["Ticker"].replace(".L", ""),
                    "Index": r["Index"],
                    "P/E": round(float(pe), 2),
                    "Latest date": four[0][0].strftime("%Y-%m-%d"),
                    "Latest EPS": vals[0],
                    "Previous EPS": vals[1],
                    "2 periods ago EPS": vals[2],
                    "3 periods ago EPS": vals[3],
                })
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["P/E", "Company"]).head(10)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    path = out / f"screen_{datetime.now():%Y-%m-%d}.csv"
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
