import math
import re
from datetime import datetime, timedelta, timezone
from io import StringIO
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import feedparser

st.set_page_config(page_title="FTSE Value + Stock Attention Screener", layout="wide")
st.title("FTSE Value + Stock Attention Screener")
st.caption("Screen FTSE 100 + FTSE 250 for low P/E shares with exactly two recent earnings increases, and compare them with the stocks receiving the most online discussion.")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152.0.0.0 Safari/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}

def _extract_constituent_table(html, index_name):
    tables = pd.read_html(StringIO(html))
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("ticker" in c or "epic" in c for c in cols) and any("company" in c for c in cols):
            ticker_col = next(c for c in t.columns if "ticker" in str(c).lower() or "epic" in str(c).lower())
            company_col = next(c for c in t.columns if "company" in str(c).lower())
            out = t[[company_col, ticker_col]].copy()
            out.columns = ["Company", "Ticker"]
            out["Index"] = index_name
            out["Ticker"] = (
                out["Ticker"].astype(str)
                .str.replace(r"\[.*?\]", "", regex=True)
                .str.strip()
                .str.replace(".", "-", regex=False)
            )
            out["Ticker"] = out["Ticker"].apply(lambda x: x if x.endswith(".L") else x + ".L")
            return out
    raise RuntimeError(f"Could not identify the {index_name} constituent table.")

@st.cache_data(ttl=60 * 60 * 12)
def get_constituents():
    sources = [
        ("FTSE 100", "https://en.wikipedia.org/wiki/FTSE_100_Index"),
        ("FTSE 250", "https://en.wikipedia.org/wiki/FTSE_250_Index"),
    ]
    frames, errors = [], []
    for index_name, url in sources:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            frames.append(_extract_constituent_table(r.text, index_name))
        except Exception as e:
            errors.append(f"{index_name}: {type(e).__name__}: {e}")
    if not frames:
        raise RuntimeError("Could not retrieve FTSE constituents. Please try again shortly. " + " | ".join(errors))
    return pd.concat(frames, ignore_index=True).dropna(subset=["Company", "Ticker"]).drop_duplicates("Ticker")

def _find_series(qf, earnings_metric):
    if qf is None or qf.empty:
        return None
    candidates = ["Diluted EPS", "Basic EPS", "Normalized EPS"] if earnings_metric == "EPS" else [
        "Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"
    ]
    for row in candidates:
        if row in qf.index:
            s = pd.to_numeric(qf.loc[row], errors="coerce").dropna()
            if len(s) >= 4:
                return s
    return None

def exactly_two_increases(values):
    if len(values) < 4:
        return False
    q0, q1, q2, q3 = values[:4]
    return q0 > q1 > q2 and not (q2 > q3)

def pct_change(new, old):
    return None if old == 0 else (new / old - 1) * 100

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def screen_stock(ticker, company, index_name, earnings_metric):
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        pe = info.get("trailingPE")
        if pe is None:
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            eps = info.get("trailingEps")
            if price and eps and eps > 0:
                pe = price / eps
        if pe is None or not isinstance(pe, (int, float)) or not math.isfinite(pe) or pe <= 0:
            return None
        s = _find_series(t.quarterly_financials, earnings_metric)
        if s is None:
            return None
        pairs = sorted([(pd.to_datetime(k), float(v)) for k, v in s.items()], reverse=True)[:4]
        vals = [v for _, v in pairs]
        if not exactly_two_increases(vals):
            return None
        return {
            "Company": company,
            "Ticker": ticker.replace(".L", ""),
            "Index": index_name,
            "P/E": round(float(pe), 2),
            "Latest period": pairs[0][0].strftime("%Y-%m-%d"),
            "Latest earnings": vals[0],
            "Previous": vals[1],
            "2 periods ago": vals[2],
            "3 periods ago": vals[3],
            "Latest growth %": pct_change(vals[0], vals[1]),
            "Prior growth %": pct_change(vals[1], vals[2]),
            "Earlier growth %": pct_change(vals[2], vals[3]),
        }
    except Exception:
        return None

def clean_company_name(name):
    x = re.sub(r"\b(plc|limited|ltd|group plc|holdings plc|holdings|group)\b", "", str(name), flags=re.I)
    return re.sub(r"\s+", " ", x).strip(" ,.-")

@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def article_activity(company, ticker, days):
    short_name = clean_company_name(company)
    ticker_plain = ticker.replace(".L", "")
    q = f'"{short_name}" OR "{ticker_plain}" stock OR shares'
    url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-GB&gl=GB&ceid=GB:en"
    try:
        resp = requests.get(url, timeout=12, headers=HEADERS)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
        seen, items = set(), []
        for e in feed.entries:
            published = getattr(e, "published_parsed", None)
            if published is None:
                continue
            dt = datetime(*published[:6], tzinfo=timezone.utc)
            if dt < cutoff:
                continue
            title = getattr(e, "title", "")
            link = getattr(e, "link", "")
            key = (title.strip().lower(), link)
            if key in seen:
                continue
            seen.add(key)
            items.append((dt, title))
        return {
            "count": len(items),
            "latest_article": max([x[0] for x in items], default=None),
            "sample_titles": [x[1] for x in items[:3]],
        }
    except Exception:
        return {"count": None, "latest_article": None, "sample_titles": []}

def add_activity_rank(df):
    if df.empty:
        return df
    x = df.copy()
    x["Articles"] = pd.to_numeric(x["Articles"], errors="coerce")
    valid = x["Articles"].notna()
    x.loc[valid, "Activity rank"] = x.loc[valid, "Articles"].rank(method="min", ascending=False)
    n = int(valid.sum())
    if n > 1:
        x.loc[valid, "Activity percentile"] = 100 * (n - x.loc[valid, "Activity rank"]) / (n - 1)
    elif n == 1:
        x.loc[valid, "Activity percentile"] = 100.0
    return x

def activity_label(p):
    if pd.isna(p): return "No data"
    if p >= 90: return "Top 10% discussion"
    if p >= 75: return "High discussion"
    if p >= 25: return "Middle discussion"
    if p >= 10: return "Low discussion"
    return "Bottom 10% discussion"

try:
    all_constituents = get_constituents()
except Exception as e:
    st.error("The app is online, but the FTSE constituent list could not be loaded. Please refresh in a few minutes.")
    with st.expander("Technical details"):
        st.code(str(e))
    st.stop()

with st.sidebar:
    st.header("Report settings")
    period_days = st.radio("Discussion period", [7, 30], format_func=lambda x: f"Last {x} days", horizontal=True)
    index_filter = st.multiselect("Indices", ["FTSE 100", "FTSE 250"], default=["FTSE 100", "FTSE 250"])
    top_n = st.number_input("Number of value-screen stocks", 1, 50, 10, 1)
    metric = st.selectbox("Earnings measure", ["EPS", "Net income"], index=0)
    max_pe = st.number_input("Maximum P/E (optional)", min_value=0.0, value=0.0, step=1.0, help="Leave at 0 for no maximum.")
    st.markdown("**Earnings rule**  \nLatest > previous > two periods ago, but two periods ago is **not** greater than three periods ago.")

constituents = all_constituents[all_constituents["Index"].isin(index_filter)].reset_index(drop=True)
tabs = st.tabs(["Combined report", "Most talked about", "Least talked about", "Value screen only", "Method"])

def run_activity_scan():
    rows, total = [], len(constituents)
    prog, status = st.progress(0), st.empty()
    for i, r in constituents.iterrows():
        status.write(f"Measuring discussion {i+1:,} of {total:,}: {r['Company']}")
        a = article_activity(r["Company"], r["Ticker"], period_days)
        rows.append({"Company": r["Company"], "Ticker": r["Ticker"].replace(".L", ""), "Index": r["Index"], "Articles": a["count"], "Latest article": a["latest_article"], "Example headlines": " | ".join(a["sample_titles"])})
        prog.progress((i + 1) / total)
    prog.empty(); status.empty()
    return add_activity_rank(pd.DataFrame(rows))

def run_value_scan():
    rows, total = [], len(constituents)
    prog, status = st.progress(0), st.empty()
    for i, r in constituents.iterrows():
        status.write(f"Checking fundamentals {i+1:,} of {total:,}: {r['Company']}")
        result = screen_stock(r["Ticker"], r["Company"], r["Index"], metric)
        if result is not None and (max_pe <= 0 or result["P/E"] <= max_pe):
            rows.append(result)
        prog.progress((i + 1) / total)
    prog.empty(); status.empty()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["P/E", "Company"]).head(int(top_n)).reset_index(drop=True)

with tabs[0]:
    st.subheader(f"Combined value + discussion report â€” last {period_days} days")
    if st.button("Run combined report", type="primary"):
        activity_df = run_activity_scan()
        value_df = run_value_scan()
        if value_df.empty:
            st.warning("No qualifying value-screen shares were found with the available data.")
        else:
            merged = value_df.merge(activity_df[["Ticker", "Articles", "Activity rank", "Activity percentile", "Latest article"]], on="Ticker", how="left")
            merged["Discussion band"] = merged["Activity percentile"].apply(activity_label)
            merged["Activity percentile"] = merged["Activity percentile"].round(1)
            st.dataframe(merged, use_container_width=True, hide_index=True)
            valid = activity_df.dropna(subset=["Articles"])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Top 10 most discussed â€” last {period_days} days**")
                st.dataframe(valid.sort_values(["Articles", "Company"], ascending=[False, True]).head(10)[["Company","Ticker","Index","Articles","Activity rank"]], hide_index=True, use_container_width=True)
            with c2:
                st.markdown(f"**Bottom 10 least discussed â€” last {period_days} days**")
                st.dataframe(valid.sort_values(["Articles", "Company"], ascending=[True, True]).head(10)[["Company","Ticker","Index","Articles","Activity rank"]], hide_index=True, use_container_width=True)
            st.download_button("Download combined CSV", merged.to_csv(index=False).encode("utf-8"), file_name=f"ftse_combined_{period_days}d_{datetime.now():%Y-%m-%d}.csv", mime="text/csv")

with tabs[1]:
    st.subheader(f"Most talked about FTSE stocks â€” last {period_days} days")
    if st.button("Run most-talked-about report"):
        a = run_activity_scan().dropna(subset=["Articles"]).sort_values(["Articles","Company"], ascending=[False, True]).head(25)
        st.dataframe(a, hide_index=True, use_container_width=True)

with tabs[2]:
    st.subheader(f"Least talked about FTSE stocks â€” last {period_days} days")
    if st.button("Run least-talked-about report"):
        a = run_activity_scan().dropna(subset=["Articles"]).sort_values(["Articles","Company"], ascending=[True, True]).head(25)
        st.dataframe(a, hide_index=True, use_container_width=True)

with tabs[3]:
    st.subheader("Low P/E + exactly two recent earnings increases")
    if st.button("Run value screen"):
        v = run_value_scan()
        st.warning("No qualifying shares were found with the available data.") if v.empty else st.dataframe(v, hide_index=True, use_container_width=True)

with tabs[4]:
    st.markdown("""
### Fundamental rule
A stock qualifies when the latest four quarterly earnings observations satisfy `Q0 > Q1 > Q2` and **NOT** `Q2 > Q3`.

### Discussion activity
The discussion score is a free estimate based on Google News RSS search results, not an exhaustive internet-wide mention count.

### Data limitations
Yahoo Finance and free news-search data can occasionally be incomplete or delayed. This is a screening/research tool rather than investment advice.
""")
