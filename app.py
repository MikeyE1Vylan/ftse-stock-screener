
import math
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import feedparser

st.set_page_config(page_title="FTSE Value + Stock Attention Screener", layout="wide")

st.title("FTSE Value + Stock Attention Screener")
st.caption(
    "Screen FTSE 100 + FTSE 250 for low P/E shares with exactly two recent earnings increases, "
    "and compare them with the stocks receiving the most online discussion."
)

# -----------------------------
# Constituents
# -----------------------------
@st.cache_data(ttl=60 * 60 * 12)
def get_constituents():
    urls = [
        ("FTSE 100", "https://en.wikipedia.org/wiki/FTSE_100_Index"),
        ("FTSE 250", "https://en.wikipedia.org/wiki/FTSE_250_Index"),
    ]
    frames = []
    for index_name, url in urls:
        tables = pd.read_html(url)
        chosen = None
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("ticker" in c or "epic" in c for c in cols) and any("company" in c for c in cols):
                chosen = t.copy()
                break
        if chosen is None:
            continue

        ticker_col = next(c for c in chosen.columns if "ticker" in str(c).lower() or "epic" in str(c).lower())
        company_col = next(c for c in chosen.columns if "company" in str(c).lower())

        out = chosen[[company_col, ticker_col]].copy()
        out.columns = ["Company", "Ticker"]
        out["Index"] = index_name
        out["Ticker"] = (
            out["Ticker"].astype(str)
            .str.replace(r"\[.*?\]", "", regex=True)
            .str.strip()
            .str.replace(".", "-", regex=False)
            + ".L"
        )
        frames.append(out)

    if not frames:
        raise RuntimeError("Could not retrieve FTSE constituent tables.")
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Ticker"])

# -----------------------------
# Fundamental screen
# -----------------------------
def _find_eps_series(qf: pd.DataFrame):
    if qf is None or qf.empty:
        return None
    for row in ["Diluted EPS", "Basic EPS", "Normalized EPS"]:
        if row in qf.index:
            s = pd.to_numeric(qf.loc[row], errors="coerce").dropna()
            if len(s) >= 4:
                return s
    return None

def _find_net_income_series(qf: pd.DataFrame):
    if qf is None or qf.empty:
        return None
    for row in [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Including Noncontrolling Interests",
    ]:
        if row in qf.index:
            s = pd.to_numeric(qf.loc[row], errors="coerce").dropna()
            if len(s) >= 4:
                return s
    return None

def _sorted_latest_first(series):
    pairs = []
    for k, v in series.items():
        try:
            dt = pd.to_datetime(k)
        except Exception:
            dt = pd.Timestamp.min
        pairs.append((dt, float(v)))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs

def exactly_two_increases(values):
    # Latest four observations q0, q1, q2, q3 (latest first)
    # Qualifies if q0 > q1 > q2, but q2 was NOT above q3.
    if len(values) < 4:
        return False
    q0, q1, q2, q3 = values[:4]
    return q0 > q1 > q2 and not (q2 > q3)

def pct_change(new, old):
    if old == 0:
        return None
    return (new / old - 1) * 100

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

        qf = t.quarterly_financials
        series = _find_eps_series(qf) if earnings_metric == "EPS" else _find_net_income_series(qf)
        if series is None:
            return None

        pairs = _sorted_latest_first(series)
        vals = [v for _, v in pairs[:4]]
        dates = [d.strftime("%Y-%m-%d") for d, _ in pairs[:4]]

        if not exactly_two_increases(vals):
            return None

        return {
            "Company": company,
            "Ticker": ticker.replace(".L", ""),
            "Index": index_name,
            "P/E": round(float(pe), 2),
            "Latest period": dates[0],
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

# -----------------------------
# Online discussion / article activity
# -----------------------------
def clean_company_name(name: str) -> str:
    # Remove common legal suffixes that can reduce search quality.
    x = str(name)
    x = re.sub(r"\b(plc|limited|ltd|group plc|holdings plc|holdings|group)\b", "", x, flags=re.I)
    x = re.sub(r"\s+", " ", x).strip(" ,.-")
    return x

def google_news_rss_url(query: str):
    # UK English Google News RSS search.
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-GB&gl=GB&ceid=GB:en"

@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def article_activity(company, ticker, days):
    """
    Free estimate based on Google News RSS search results.
    This is NOT a complete count of all online discussion.
    We search company name plus London-share/ticker context and count unique RSS items
    whose publication dates fall inside the selected window.
    """
    short_name = clean_company_name(company)
    ticker_plain = ticker.replace(".L", "")

    # Search terms designed to reduce false positives.
    query = f'"{short_name}" OR "{ticker_plain}" stock OR shares'
    url = google_news_rss_url(query)

    try:
        resp = requests.get(
            url,
            timeout=12,
            headers={"User-Agent": "Mozilla/5.0 FTSE-screen/1.0"}
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
        seen = set()
        items = []

        for e in feed.entries:
            title = getattr(e, "title", "")
            link = getattr(e, "link", "")
            published = getattr(e, "published_parsed", None)

            if published is None:
                continue
            dt = datetime(*published[:6], tzinfo=timezone.utc)
            if dt < cutoff:
                continue

            key = (title.strip().lower(), link)
            if key in seen:
                continue
            seen.add(key)

            source = ""
            if hasattr(e, "source") and isinstance(e.source, dict):
                source = e.source.get("title", "")

            items.append({
                "title": title,
                "link": link,
                "published": dt,
                "source": source,
            })

        return {
            "count": len(items),
            "latest_article": max([x["published"] for x in items], default=None),
            "sample_titles": [x["title"] for x in items[:3]],
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
        # 100 = most discussed, 0 = least discussed.
        x.loc[valid, "Activity percentile"] = (
            100 * (n - x.loc[valid, "Activity rank"]) / (n - 1)
        )
    else:
        x.loc[valid, "Activity percentile"] = 100.0
    return x

def activity_label(percentile):
    if pd.isna(percentile):
        return "No data"
    if percentile >= 90:
        return "Top 10% discussion"
    if percentile >= 75:
        return "High discussion"
    if percentile >= 25:
        return "Middle discussion"
    if percentile >= 10:
        return "Low discussion"
    return "Bottom 10% discussion"

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Report settings")
    period_days = st.radio("Discussion period", [7, 30], format_func=lambda x: f"Last {x} days", horizontal=True)
    index_filter = st.multiselect("Indices", ["FTSE 100", "FTSE 250"], default=["FTSE 100", "FTSE 250"])
    top_n = st.number_input("Number of value-screen stocks", min_value=1, max_value=50, value=10, step=1)
    metric = st.selectbox("Earnings measure", ["EPS", "Net income"], index=0)
    max_pe = st.number_input(
        "Maximum P/E (optional)", min_value=0.0, value=0.0, step=1.0,
        help="Leave at 0 for no maximum."
    )
    st.markdown(
        "**Earnings rule**  \n"
        "Latest > previous > two periods ago, but two periods ago is **not** greater "
        "than three periods ago."
    )

tabs = st.tabs([
    "Combined report",
    "Most talked about",
    "Least talked about",
    "Value screen only",
    "Method"
])

constituents = get_constituents()
constituents = constituents[constituents["Index"].isin(index_filter)].reset_index(drop=True)

# -----------------------------
# Shared runner
# -----------------------------
def run_activity_scan():
    rows = []
    total = len(constituents)
    prog = st.progress(0)
    status = st.empty()
    for i, r in constituents.iterrows():
        status.write(f"Measuring discussion {i+1:,} of {total:,}: {r['Company']}")
        a = article_activity(r["Company"], r["Ticker"], period_days)
        rows.append({
            "Company": r["Company"],
            "Ticker": r["Ticker"].replace(".L", ""),
            "Index": r["Index"],
            "Articles": a["count"],
            "Latest article": a["latest_article"],
            "Example headlines": " | ".join(a["sample_titles"]),
        })
        prog.progress((i + 1) / total)
    prog.empty()
    status.empty()
    return add_activity_rank(pd.DataFrame(rows))

def run_value_scan():
    rows = []
    total = len(constituents)
    prog = st.progress(0)
    status = st.empty()
    for i, r in constituents.iterrows():
        status.write(f"Checking fundamentals {i+1:,} of {total:,}: {r['Company']}")
        result = screen_stock(r["Ticker"], r["Company"], r["Index"], metric)
        if result is not None and (max_pe <= 0 or result["P/E"] <= max_pe):
            rows.append(result)
        prog.progress((i + 1) / total)
    prog.empty()
    status.empty()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["P/E", "Company"]).head(int(top_n)).reset_index(drop=True)

# -----------------------------
# Combined
# -----------------------------
with tabs[0]:
    st.subheader(f"Combined value + discussion report — last {period_days} days")
    st.write(
        "Runs both screens and shows where every low-P/E / earnings-growth qualifier sits "
        "within the full FTSE discussion ranking."
    )
    if st.button("Run combined report", type="primary", key="run_combined"):
        activity_df = run_activity_scan()
        value_df = run_value_scan()

        if value_df.empty:
            st.warning("No qualifying value-screen shares were found with the available data.")
        else:
            merged = value_df.merge(
                activity_df[
                    ["Ticker", "Articles", "Activity rank", "Activity percentile", "Latest article"]
                ],
                on="Ticker",
                how="left"
            )
            merged["Discussion band"] = merged["Activity percentile"].apply(activity_label)
            merged["Activity rank"] = merged["Activity rank"].astype("Int64")
            merged["Activity percentile"] = merged["Activity percentile"].round(1)

            st.dataframe(
                merged.style.format({
                    "P/E": "{:.2f}",
                    "Latest growth %": lambda x: "" if pd.isna(x) else f"{x:.1f}%",
                    "Prior growth %": lambda x: "" if pd.isna(x) else f"{x:.1f}%",
                    "Earlier growth %": lambda x: "" if pd.isna(x) else f"{x:.1f}%",
                    "Activity percentile": lambda x: "" if pd.isna(x) else f"{x:.1f}",
                }),
                use_container_width=True
            )

            # Context tables
            valid_activity = activity_df.dropna(subset=["Articles"]).copy()
            top10 = valid_activity.sort_values(["Articles", "Company"], ascending=[False, True]).head(10)
            bottom10 = valid_activity.sort_values(["Articles", "Company"], ascending=[True, True]).head(10)

            st.markdown("#### Discussion context")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Top 10 most discussed — last {period_days} days**")
                st.dataframe(
                    top10[["Company", "Ticker", "Index", "Articles", "Activity rank"]],
                    hide_index=True,
                    use_container_width=True
                )
            with c2:
                st.markdown(f"**Bottom 10 least discussed — last {period_days} days**")
                st.dataframe(
                    bottom10[["Company", "Ticker", "Index", "Articles", "Activity rank"]],
                    hide_index=True,
                    use_container_width=True
                )

            csv = merged.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download combined CSV",
                csv,
                file_name=f"ftse_combined_{period_days}d_{datetime.now():%Y-%m-%d}.csv",
                mime="text/csv",
                key="combined_csv"
            )

# -----------------------------
# Most discussed
# -----------------------------
with tabs[1]:
    st.subheader(f"Most talked about FTSE stocks — last {period_days} days")
    if st.button("Run most-talked-about report", key="run_top_activity"):
        activity_df = run_activity_scan()
        top = activity_df.dropna(subset=["Articles"]).sort_values(
            ["Articles", "Company"], ascending=[False, True]
        ).head(25)
        st.dataframe(
            top[["Company", "Ticker", "Index", "Articles", "Activity rank", "Activity percentile", "Latest article", "Example headlines"]],
            hide_index=True,
            use_container_width=True
        )
        st.download_button(
            "Download report CSV",
            top.to_csv(index=False).encode("utf-8"),
            file_name=f"ftse_most_discussed_{period_days}d_{datetime.now():%Y-%m-%d}.csv",
            mime="text/csv",
            key="top_csv"
        )

# -----------------------------
# Least discussed
# -----------------------------
with tabs[2]:
    st.subheader(f"Least talked about FTSE stocks — last {period_days} days")
    if st.button("Run least-talked-about report", key="run_bottom_activity"):
        activity_df = run_activity_scan()
        bottom = activity_df.dropna(subset=["Articles"]).sort_values(
            ["Articles", "Company"], ascending=[True, True]
        ).head(25)
        st.dataframe(
            bottom[["Company", "Ticker", "Index", "Articles", "Activity rank", "Activity percentile", "Latest article", "Example headlines"]],
            hide_index=True,
            use_container_width=True
        )
        st.download_button(
            "Download report CSV",
            bottom.to_csv(index=False).encode("utf-8"),
            file_name=f"ftse_least_discussed_{period_days}d_{datetime.now():%Y-%m-%d}.csv",
            mime="text/csv",
            key="bottom_csv"
        )

# -----------------------------
# Value only
# -----------------------------
with tabs[3]:
    st.subheader("Low P/E + exactly two recent earnings increases")
    if st.button("Run value screen", key="run_value_only"):
        value_df = run_value_scan()
        if value_df.empty:
            st.warning("No qualifying shares were found with the available data.")
        else:
            st.dataframe(value_df, hide_index=True, use_container_width=True)
            st.download_button(
                "Download value CSV",
                value_df.to_csv(index=False).encode("utf-8"),
                file_name=f"ftse_value_screen_{datetime.now():%Y-%m-%d}.csv",
                mime="text/csv",
                key="value_csv"
            )

# -----------------------------
# Method
# -----------------------------
with tabs[4]:
    st.markdown(
        f"""
### Fundamental rule
A stock qualifies when its latest four quarterly earnings observations satisfy:

`Q0 > Q1 > Q2` and **NOT** `Q2 > Q3`

That means earnings have increased across exactly the latest two transitions, rather than
already being on a three-or-more-period growth run.

### Discussion activity
The discussion score is a **free estimate of online news/article activity**, not a complete
internet-wide mention count. The app searches Google News RSS for each FTSE company and counts
unique returned articles published within the selected **7-day or 30-day** window.

For each stock the app calculates:
- article count in the selected window;
- rank versus all selected FTSE constituents;
- discussion percentile, where 100 is the most discussed;
- a discussion band from Bottom 10% through Top 10%;
- sample recent headlines.

### Why this is useful
The combined report lets you see whether a low-P/E stock with newly improving earnings is:
- already attracting unusually high attention;
- around the middle of the market;
- or still receiving very little discussion.

### Limitations
Free RSS search results are incomplete and can be capped or affected by company-name ambiguity.
They are best treated as a consistent **relative activity indicator**, rather than an audited count
of every article on the internet. Fundamental data from Yahoo Finance can also be delayed or missing.

This tool is for research/screening rather than investment advice.
"""
    )
