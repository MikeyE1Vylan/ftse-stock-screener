
import time
import math
import re
from datetime import datetime, timedelta, timezone
from io import StringIO
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="FTSE Value + Stock Attention Screener",
    layout="wide"
)

st.title("FTSE Value + Stock Attention Screener")
st.caption("VERSION 10 VERIFIED — FRESH SCAN + ROBUST TRAILING P/E + TOP 10")
st.caption(
    "FTSE 100 + FTSE 250 operating companies • trusts/funds/ETFs excluded • lowest P/E • "
    "7-day / 30-day online discussion activity"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/152.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}

# ============================================================
# FTSE CONSTITUENTS
# ============================================================

def _extract_constituent_table(html, index_name):
    tables = pd.read_html(StringIO(html))

    for table in tables:
        cols = [str(c).lower() for c in table.columns]
        has_ticker = any("ticker" in c or "epic" in c for c in cols)
        has_company = any("company" in c for c in cols)

        if not (has_ticker and has_company):
            continue

        ticker_col = next(
            c for c in table.columns
            if "ticker" in str(c).lower() or "epic" in str(c).lower()
        )
        company_col = next(
            c for c in table.columns
            if "company" in str(c).lower()
        )

        sector_col = None
        for c in table.columns:
            cl = str(c).lower()
            if "sector" in cl or "industry" in cl or "supersector" in cl:
                sector_col = c
                break

        keep_cols = [company_col, ticker_col]
        if sector_col is not None:
            keep_cols.append(sector_col)

        out = table[keep_cols].copy()

        if sector_col is not None:
            out.columns = ["Company", "Ticker", "Sector"]
        else:
            out.columns = ["Company", "Ticker"]
            out["Sector"] = ""

        out["Index"] = index_name

        out["Ticker"] = (
            out["Ticker"]
            .astype(str)
            .str.replace(r"\[.*?\]", "", regex=True)
            .str.strip()
            .str.replace(".", "-", regex=False)
        )

        out["Ticker"] = out["Ticker"].apply(
            lambda x: x if x.endswith(".L") else x + ".L"
        )

        return out

    raise RuntimeError(f"Could not identify constituent table for {index_name}.")


INVESTMENT_VEHICLE_TICKERS = {
    "PCT.L", "ATT.L", "PSH.L", "FCIT.L", "SMT.L", "MNKS.L", "JGGI.L",
    "BRGE.L", "BRSC.L", "THRG.L", "HRI.L", "USA.L", "PHI.L", "TEM.L",
    "JMG.L", "IEM.L", "HICL.L", "INPP.L", "3IN.L", "BBGI.L", "TRIG.L",
    "UKW.L", "FSFL.L", "BSIF.L", "NESF.L", "GCP.L", "SEQI.L", "RCP.L",
    "PIN.L", "HVPE.L", "ICGT.L", "NBPE.L", "OCI.L", "APAX.L", "CTY.L",
    "LWDB.L", "MRCH.L", "EDIN.L", "MRC.L", "JAM.L", "SAIN.L", "BNKR.L",
    "BUT.L", "LTI.L", "SDP.L", "HFEL.L", "AAIF.L", "SOI.L", "JII.L",
    "BGCG.L", "FSG.L", "SSON.L", "AAS.L", "ATR.L", "STS.L", "MUT.L",
    "MYI.L", "DIVI.L", "TMPL.L", "CHRY.L", "AUGM.L", "PINT.L", "CORD.L",
    "DGI9.L"
}

def is_investment_vehicle(company, sector="", ticker=""):
    name = str(company).lower().strip()
    sector_text = str(sector).lower().strip()
    ticker_text = str(ticker).upper().strip()
    if ticker_text in INVESTMENT_VEHICLE_TICKERS:
        return True
    name_terms = [
        "investment trust", "technology trust", "income trust", "venture capital trust",
        "infrastructure fund", "income fund", "solar fund", "closed-ended fund",
        "closed end fund", "closed-end fund", "private equity partners",
        " ucits", " etf", " etc", "exchange traded fund", "exchange-traded fund"
    ]
    if any(x in name for x in name_terms):
        return True
    sector_terms = [
        "closed end investments", "closed-end investments", "closed ended investments",
        "closed-ended investments", "investment trust", "investment trusts",
        "investment fund", "investment funds", "collective investments",
        "exchange traded fund", "exchange-traded fund", "etf"
    ]
    return any(x in sector_text for x in sector_terms)


@st.cache_data(ttl=60 * 60 * 12)
def get_constituents():
    sources = [
        ("FTSE 100", "https://en.wikipedia.org/wiki/FTSE_100_Index"),
        ("FTSE 250", "https://en.wikipedia.org/wiki/FTSE_250_Index"),
    ]

    frames = []
    errors = []

    for index_name, url in sources:
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            frames.append(_extract_constituent_table(response.text, index_name))
        except Exception as exc:
            errors.append(f"{index_name}: {type(exc).__name__}: {exc}")

    if not frames:
        raise RuntimeError(
            "Could not retrieve FTSE constituents. " + " | ".join(errors)
        )

    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["Company", "Ticker"])
    result = result.drop_duplicates(subset=["Ticker"])

    result["Excluded investment vehicle"] = result.apply(
        lambda r: is_investment_vehicle(r["Company"], r.get("Sector", ""), r["Ticker"]),
        axis=1,
    )
    result = result[result["Excluded investment vehicle"] == False].copy()

    return result


# ============================================================
# EARNINGS DATA
# ============================================================

def _normalise_date(value):
    try:
        return pd.to_datetime(value).tz_localize(None)
    except Exception:
        try:
            return pd.to_datetime(value)
        except Exception:
            return pd.NaT


def _collect_reported_eps_from_earnings_dates(ticker_obj):
    rows = []

    try:
        ed = ticker_obj.get_earnings_dates(limit=16)
    except Exception:
        try:
            ed = ticker_obj.earnings_dates
        except Exception:
            ed = None

    if ed is None or len(ed) == 0:
        return rows

    df = ed.copy()

    if "Reported EPS" not in df.columns:
        return rows

    for idx, row in df.iterrows():
        eps = pd.to_numeric(row.get("Reported EPS"), errors="coerce")
        if pd.isna(eps):
            continue

        dt = _normalise_date(idx)
        if pd.isna(dt):
            continue

        rows.append({
            "date": dt,
            "value": float(eps),
            "source": "Reported EPS",
        })

    return rows


def _collect_eps_from_statement(statement, source_name):
    rows = []

    if statement is None or statement.empty:
        return rows

    row_names = [
        "Diluted EPS",
        "Basic EPS",
        "Normalized EPS",
    ]

    selected = None
    for name in row_names:
        if name in statement.index:
            selected = statement.loc[name]
            break

    if selected is None:
        return rows

    selected = pd.to_numeric(selected, errors="coerce").dropna()

    for col, value in selected.items():
        dt = _normalise_date(col)
        if pd.isna(dt):
            continue

        rows.append({
            "date": dt,
            "value": float(value),
            "source": source_name,
        })

    return rows


def _dedupe_and_sort(items):
    if not items:
        return []
    items = sorted(items, key=lambda x: x["date"], reverse=True)
    result = []
    for item in items:
        if any(abs((item["date"] - x["date"]).days) <= 14 for x in result):
            continue
        result.append(item)
    return result


def _infer_frequency(items):
    items = _dedupe_and_sort(items)
    if len(items) < 4:
        return None
    gaps = [abs((a["date"] - b["date"]).days) for a, b in zip(items[:4], items[1:4])]
    median_gap = sorted(gaps)[len(gaps) // 2]
    if median_gap <= 140:
        return "Quarterly"
    if median_gap <= 230:
        return "Half-year"
    return "Annual"


def get_latest_four_earnings(ticker_obj):
    """
    Four like-for-like EPS observations only.
    Priority: reported EPS history, quarterly statement EPS, annual statement EPS.
    Frequencies are never mixed.
    """
    reported = _dedupe_and_sort(_collect_reported_eps_from_earnings_dates(ticker_obj))
    if len(reported) >= 4:
        frequency = _infer_frequency(reported) or "Reported periods"
        chosen = reported[:4]
        for x in chosen:
            x["frequency"] = frequency
        return chosen

    try:
        quarterly = _dedupe_and_sort(
            _collect_eps_from_statement(ticker_obj.quarterly_financials, "Quarterly statement EPS")
        )
    except Exception:
        quarterly = []

    if len(quarterly) >= 4:
        chosen = quarterly[:4]
        for x in chosen:
            x["frequency"] = "Quarterly"
        return chosen

    try:
        annual = _dedupe_and_sort(
            _collect_eps_from_statement(ticker_obj.financials, "Annual statement EPS")
        )
    except Exception:
        annual = []

    if len(annual) >= 4:
        chosen = annual[:4]
        for x in chosen:
            x["frequency"] = "Annual"
        return chosen

    candidates = [("Quarterly", quarterly), ("Reported periods", reported), ("Annual", annual)]
    frequency, best = max(candidates, key=lambda pair: len(pair[1]))
    chosen = best[:4]
    for x in chosen:
        x["frequency"] = frequency
    return chosen

def classify_earnings(items):
    if len(items) < 4:
        return "Insufficient earnings history"

    q0, q1, q2, q3 = [x["value"] for x in items[:4]]

    if q0 > q1 > q2 and not (q2 > q3):
        return "PASS — exactly last 2 increased"

    if q0 > q1 > q2 and q2 > q3:
        return "3+ consecutive increases"

    if q0 > q1 and not (q1 > q2):
        return "Only latest period increased"

    if not (q0 > q1):
        return "Latest period did not increase"

    return "Does not meet exact rule"


def pct_change(new, old):
    try:
        if old == 0:
            return None
        return (new / old - 1) * 100
    except Exception:
        return None


# ============================================================
# P/E CALCULATION
# ============================================================

def _safe_fast_info(ticker_obj):
    try:
        return ticker_obj.fast_info
    except Exception:
        return {}


def _safe_info(ticker_obj):
    try:
        return ticker_obj.get_info() or {}
    except Exception:
        try:
            return ticker_obj.info or {}
        except Exception:
            return {}


def _latest_positive_net_income(statement):
    if statement is None or statement.empty:
        return None

    candidates = [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Including Noncontrolling Interests",
    ]

    for row_name in candidates:
        if row_name in statement.index:
            s = pd.to_numeric(statement.loc[row_name], errors="coerce").dropna()

            if len(s):
                try:
                    latest = float(
                        sorted(
                            [(pd.to_datetime(k), v) for k, v in s.items()],
                            key=lambda x: x[0],
                            reverse=True,
                        )[0][1]
                    )
                    return latest
                except Exception:
                    return float(s.iloc[0])

    return None


def _positive_number(value):
    try:
        value = float(value)
        if math.isfinite(value) and value > 0:
            return value
    except Exception:
        pass
    return None


def _latest_close(ticker_obj):
    """Get a recent unadjusted close without relying on the info endpoint."""
    try:
        hist = ticker_obj.history(
            period="5d",
            interval="1d",
            auto_adjust=False,
            repair=True,
        )
        if hist is not None and not hist.empty and "Close" in hist.columns:
            closes = pd.to_numeric(hist["Close"], errors="coerce").dropna()
            if len(closes):
                return _positive_number(closes.iloc[-1])
    except Exception:
        pass

    fast = _safe_fast_info(ticker_obj)
    for key in ("last_price", "regular_market_price"):
        try:
            value = fast.get(key)
        except Exception:
            try:
                value = fast[key]
            except Exception:
                value = None
        value = _positive_number(value)
        if value:
            return value

    return None


def _ttm_net_income(ticker_obj):
    """
    Prefer Yahoo's trailing income statement. If unavailable, sum the latest
    four quarterly net-income observations. This is a true TTM fallback.
    """
    candidates = [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Including Noncontrolling Interests",
    ]

    try:
        stmt = ticker_obj.get_income_stmt(freq="trailing")
        if stmt is not None and not stmt.empty:
            for row in candidates:
                if row in stmt.index:
                    vals = pd.to_numeric(stmt.loc[row], errors="coerce").dropna()
                    if len(vals):
                        return float(vals.iloc[0])
    except Exception:
        pass

    try:
        stmt = ticker_obj.get_income_stmt(freq="quarterly")
        if stmt is not None and not stmt.empty:
            for row in candidates:
                if row in stmt.index:
                    s = pd.to_numeric(stmt.loc[row], errors="coerce").dropna()
                    if len(s) >= 4:
                        dated = []
                        for k, v in s.items():
                            try:
                                dated.append((pd.to_datetime(k), float(v)))
                            except Exception:
                                pass
                        dated.sort(key=lambda x: x[0], reverse=True)
                        if len(dated) >= 4:
                            return sum(v for _, v in dated[:4])
    except Exception:
        pass

    return None


def _shares_outstanding(ticker_obj, info=None):
    """Retrieve current/recent shares outstanding from several Yahoo paths."""
    fast = _safe_fast_info(ticker_obj)
    for key in ("shares", "shares_outstanding"):
        try:
            value = fast.get(key)
        except Exception:
            try:
                value = fast[key]
            except Exception:
                value = None
        value = _positive_number(value)
        if value:
            return value

    if info:
        value = _positive_number(info.get("sharesOutstanding"))
        if value:
            return value

    try:
        shares = ticker_obj.get_shares_full(
            start=(datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
        )
        if shares is not None and len(shares):
            vals = pd.to_numeric(shares, errors="coerce").dropna()
            if len(vals):
                return _positive_number(vals.iloc[-1])
    except Exception:
        pass

    return None


def calculate_pe(ticker_obj):
    """
    Return (positive trailing P/E, method).

    Hierarchy:
      1. Yahoo current valuation-measures P/E.
      2. Yahoo info trailingPE.
      3. Current price / trailingEps.
      4. Market cap / true TTM net income.
      5. Current price / (TTM net income / shares outstanding).

    Annual net income is deliberately NOT used as a trailing-P/E substitute.
    """
    # 1) Valuation endpoint — current P/E without depending on get_info().
    try:
        valuation = ticker_obj.get_valuation_measures(freq="trailing", periods=0)
        if valuation is not None and not valuation.empty:
            for label in (
                "Trailing P/E",
                "Trailing PE",
                "Price/Earnings",
                "P/E",
                "Pe Ratio",
            ):
                if label in valuation.index and "Current" in valuation.columns:
                    pe = _positive_number(valuation.loc[label, "Current"])
                    if pe:
                        return pe, "Yahoo valuation trailing P/E"
    except Exception:
        pass

    info = _safe_info(ticker_obj)

    # 2) Standard Yahoo trailing P/E.
    pe = _positive_number(info.get("trailingPE"))
    if pe:
        return pe, "Yahoo info trailing P/E"

    # 3) Price divided by Yahoo trailing EPS.
    price = _latest_close(ticker_obj)
    trailing_eps = _positive_number(info.get("trailingEps"))
    if price and trailing_eps:
        pe = price / trailing_eps
        if math.isfinite(pe) and pe > 0:
            return pe, "Price / Yahoo trailing EPS"

    # 4) Market cap divided by true trailing-twelve-month net income.
    fast = _safe_fast_info(ticker_obj)
    market_cap = None
    try:
        market_cap = fast.get("market_cap")
    except Exception:
        try:
            market_cap = fast["market_cap"]
        except Exception:
            pass
    market_cap = _positive_number(market_cap) or _positive_number(info.get("marketCap"))

    ttm_net_income = _ttm_net_income(ticker_obj)
    if market_cap and ttm_net_income and ttm_net_income > 0:
        pe = market_cap / ttm_net_income
        if math.isfinite(pe) and pe > 0:
            return pe, "Market cap / TTM net income"

    # 5) Reconstruct TTM EPS from net income and shares, then divide price.
    shares = _shares_outstanding(ticker_obj, info)
    if price and shares and ttm_net_income and ttm_net_income > 0:
        ttm_eps = ttm_net_income / shares
        if ttm_eps > 0:
            pe = price / ttm_eps
            if math.isfinite(pe) and pe > 0:
                return pe, "Price / reconstructed TTM EPS"

    return None, None


# ============================================================
# STOCK FUNDAMENTAL ROW
# ============================================================

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_stock_fundamentals(ticker, company, index_name, scan_id=None):
    try:
        obj = yf.Ticker(ticker)

        pe, pe_method = calculate_pe(obj)

        if pe is None:
            return {
                "Company": company,
                "Ticker": ticker.replace(".L", ""),
                "Index": index_name,
                "P/E": None,
                "P/E method": "Unavailable",
                "Earnings rule": "P/E unavailable",
                "_has_pe": False,
            }

        earnings = get_latest_four_earnings(obj)
        status = classify_earnings(earnings)

        values = [x["value"] for x in earnings]
        dates = [
            x["date"].strftime("%Y-%m-%d")
            for x in earnings
        ]
        sources = [x["source"] for x in earnings]
        frequencies = [x.get("frequency") for x in earnings if x.get("frequency")]
        earnings_frequency = frequencies[0] if frequencies else "Insufficient data"

        while len(values) < 4:
            values.append(None)
            dates.append(None)
            sources.append(None)

        return {
            "Company": company,
            "Ticker": ticker.replace(".L", ""),
            "Index": index_name,
            "P/E": round(float(pe), 2),
            "P/E method": pe_method,
            "Earnings frequency": earnings_frequency,
            "Earnings rule": status,
            "Latest date": dates[0],
            "Latest EPS": values[0],
            "Previous date": dates[1],
            "Previous EPS": values[1],
            "Period -2 date": dates[2],
            "Period -2 EPS": values[2],
            "Period -3 date": dates[3],
            "Period -3 EPS": values[3],
            "Latest change %": pct_change(values[0], values[1])
                if values[0] is not None and values[1] is not None else None,
            "Previous change %": pct_change(values[1], values[2])
                if values[1] is not None and values[2] is not None else None,
            "Earnings data source": " / ".join(
                dict.fromkeys([x for x in sources if x])
            ),
            "_has_pe": True,
        }

    except Exception:
        return {
            "Company": company,
            "Ticker": ticker.replace(".L", ""),
            "Index": index_name,
            "P/E": None,
            "P/E method": "Error",
            "Earnings rule": "Data error",
            "_has_pe": False,
        }


# ============================================================
# NEWS / DISCUSSION ACTIVITY
# ============================================================

def clean_company_name(name):
    text = str(name)

    text = re.sub(
        r"\b(plc|limited|ltd|group plc|holdings plc|holdings|group)\b",
        "",
        text,
        flags=re.I,
    )

    return re.sub(r"\s+", " ", text).strip(" ,.-")


def google_news_rss_url(query):
    return (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl=en-GB&gl=GB&ceid=GB:en"
    )


@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def article_activity(company, ticker, days):
    short_name = clean_company_name(company)
    ticker_plain = ticker.replace(".L", "")

    query = f'"{short_name}" shares OR stock OR "{ticker_plain}"'

    try:
        response = requests.get(
            google_news_rss_url(query),
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()

        feed = feedparser.parse(response.content)

        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(days=int(days))
        )

        seen = set()
        items = []

        for entry in feed.entries:
            title = getattr(entry, "title", "")
            link = getattr(entry, "link", "")
            published = getattr(entry, "published_parsed", None)

            if published is None:
                continue

            dt = datetime(
                *published[:6],
                tzinfo=timezone.utc
            )

            if dt < cutoff:
                continue

            key = (
                title.strip().lower(),
                link,
            )

            if key in seen:
                continue

            seen.add(key)

            items.append({
                "title": title,
                "date": dt,
            })

        return {
            "count": len(items),
            "latest": max(
                [x["date"] for x in items],
                default=None
            ),
            "headlines": [
                x["title"]
                for x in items[:3]
            ],
        }

    except Exception:
        return {
            "count": None,
            "latest": None,
            "headlines": [],
        }


def add_activity_rank(df):
    if df.empty:
        return df

    result = df.copy()

    result["Articles"] = pd.to_numeric(
        result["Articles"],
        errors="coerce"
    )

    valid = result["Articles"].notna()

    result.loc[valid, "Activity rank"] = (
        result.loc[valid, "Articles"]
        .rank(method="min", ascending=False)
    )

    n = int(valid.sum())

    if n > 1:
        result.loc[valid, "Activity percentile"] = (
            100
            * (
                n
                - result.loc[valid, "Activity rank"]
            )
            / (n - 1)
        )

    elif n == 1:
        result.loc[valid, "Activity percentile"] = 100.0

    return result


def activity_band(value):
    if pd.isna(value):
        return "No data"

    if value >= 90:
        return "Top 10% discussion"

    if value >= 75:
        return "High discussion"

    if value >= 25:
        return "Middle discussion"

    if value >= 10:
        return "Low discussion"

    return "Bottom 10% discussion"


# ============================================================
# LOAD UNIVERSE
# ============================================================

try:
    all_constituents = get_constituents()
except Exception as exc:
    st.error(
        "The FTSE constituent list could not be loaded. "
        "Please refresh the app and try again."
    )

    with st.expander("Technical details"):
        st.code(str(exc))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Report settings")

    discussion_days = st.radio(
        "Discussion period",
        [7, 30],
        format_func=lambda x: f"Last {x} days",
        horizontal=True,
    )

    index_filter = st.multiselect(
        "Indices",
        ["FTSE 100", "FTSE 250"],
        default=["FTSE 100", "FTSE 250"],
    )

    result_count = st.number_input(
        "Number of value stocks",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )

    max_pe = st.number_input(
        "Maximum P/E (optional)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Leave at 0 for no maximum.",
    )

    st.markdown(
        """
**Important**

The **7 / 30 day setting applies only to news discussion**.

The earnings test uses **four comparable reporting periods at the same frequency**. Quarterly, half-year and annual observations are never mixed.
"""
    )


constituents = all_constituents[
    all_constituents["Index"].isin(index_filter)
].reset_index(drop=True)

st.sidebar.metric("Operating-company universe", len(constituents))
st.sidebar.caption("Investment trusts, funds, ETFs/ETCs and similar collective investment vehicles are excluded before all screens.")


# ============================================================
# RUNNERS
# ============================================================

def run_fundamental_scan(scan_id):
    rows = []

    total = len(constituents)

    progress = st.progress(0)
    status = st.empty()

    for i, row in constituents.iterrows():
        status.write(
            f"Checking fundamentals {i + 1:,} of {total:,}: "
            f"{row['Company']}"
        )

        result = get_stock_fundamentals(
            row["Ticker"],
            row["Company"],
            row["Index"],
            scan_id=scan_id,
        )

        rows.append(result)

        progress.progress((i + 1) / total)

    progress.empty()
    status.empty()

    raw = pd.DataFrame(rows)

    pe_available = raw[
        raw["_has_pe"] == True
    ].copy()

    pe_available["P/E"] = pd.to_numeric(
        pe_available["P/E"],
        errors="coerce"
    )

    pe_available = pe_available.dropna(subset=["P/E"])

    if max_pe > 0:
        pe_available = pe_available[
            pe_available["P/E"] <= max_pe
        ]

    pe_available = pe_available.sort_values(
        ["P/E", "Company"]
    )

    result = pe_available.head(
        int(result_count)
    ).copy()

    diagnostics = {
        "constituents": len(raw),
        "pe_available": len(
            raw[raw["_has_pe"] == True]
        ),
        "four_period_eps": len(raw[raw["Earnings rule"] != "Insufficient earnings history"]),
        "exact_matches": len(raw[raw["Earnings rule"] == "PASS — exactly last 2 increased"]),
        "returned": len(result),
    }

    result = result.drop(
        columns=["_has_pe"],
        errors="ignore"
    )

    return result, diagnostics, raw


def run_activity_scan():
    rows = []

    total = len(constituents)

    progress = st.progress(0)
    status = st.empty()

    for i, row in constituents.iterrows():
        status.write(
            f"Measuring discussion {i + 1:,} of {total:,}: "
            f"{row['Company']}"
        )

        activity = article_activity(
            row["Company"],
            row["Ticker"],
            discussion_days,
        )

        rows.append({
            "Company": row["Company"],
            "Ticker": row["Ticker"].replace(".L", ""),
            "Index": row["Index"],
            "Articles": activity["count"],
            "Latest article": activity["latest"],
            "Example headlines": " | ".join(
                activity["headlines"]
            ),
        })

        progress.progress((i + 1) / total)

    progress.empty()
    status.empty()

    return add_activity_rank(
        pd.DataFrame(rows)
    )


def show_diagnostics(diag):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Operating companies checked", diag["constituents"])
    c2.metric("Positive P/E available", diag["pe_available"])
    c3.metric("4 comparable EPS periods", diag["four_period_eps"])
    c4.metric("Exact earnings matches", diag["exact_matches"])
    c5.metric("Main rows returned", diag["returned"])


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "Combined report",
    "Value screen",
    "Most talked about",
    "Least talked about",
    "Method",
])


# ----------------------------
# COMBINED
# ----------------------------

with tabs[0]:
    st.subheader(
        f"Combined value + discussion report — "
        f"last {discussion_days} days"
    )

    st.write(
        "The table is ranked by lowest P/E. "
        "The earnings test uses each company's four most recent "
        "available earnings reports regardless of date."
    )

    if st.button(
        "Run combined report",
        type="primary",
        key="combined_button",
    ):
        value_df, diag, _ = run_fundamental_scan(scan_id=time.time_ns())

        show_diagnostics(diag)

        if value_df.empty:
            st.warning(
                "No shares with usable positive P/E data were returned. "
                "See the diagnostic figures above."
            )
        else:
            activity_df = run_activity_scan()

            merged = value_df.merge(
                activity_df[
                    [
                        "Ticker",
                        "Articles",
                        "Activity rank",
                        "Activity percentile",
                        "Latest article",
                    ]
                ],
                on="Ticker",
                how="left",
            )

            merged["Discussion band"] = (
                merged["Activity percentile"]
                .apply(activity_band)
            )

            merged["Activity rank"] = (
                merged["Activity rank"]
                .astype("Int64")
            )

            merged["Activity percentile"] = (
                merged["Activity percentile"]
                .round(1)
            )

            st.dataframe(
                merged,
                hide_index=True,
                use_container_width=True,
            )

            st.download_button(
                "Download combined CSV",
                merged.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    f"ftse_combined_"
                    f"{discussion_days}d_"
                    f"{datetime.now():%Y-%m-%d}.csv"
                ),
                mime="text/csv",
            )


# ----------------------------
# VALUE SCREEN
# ----------------------------

with tabs[1]:
    st.subheader(
        "Lowest P/E shares + earnings-status check"
    )

    st.write(
        "This report ALWAYS ranks the shares with usable positive "
        "P/E data from lowest to highest. The earnings-status column "
        "shows whether each stock passes your exact two-period rule."
    )

    if st.button(
        "Run value screen",
        key="value_button",
    ):
        value_df, diag, raw = run_fundamental_scan(scan_id=time.time_ns())

        show_diagnostics(diag)

        if value_df.empty:
            st.warning(
                "No shares with usable positive P/E data were returned."
            )
        else:
            st.dataframe(
                value_df,
                hide_index=True,
                use_container_width=True,
            )

            st.download_button(
                "Download value report CSV",
                value_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    f"ftse_value_"
                    f"{datetime.now():%Y-%m-%d}.csv"
                ),
                mime="text/csv",
            )

        st.markdown("---")
        st.subheader("All exact two-period earnings matches")
        exact_df = raw[
            (raw["_has_pe"] == True)
            & (raw["Earnings rule"] == "PASS — exactly last 2 increased")
        ].copy()
        exact_df["P/E"] = pd.to_numeric(exact_df["P/E"], errors="coerce")
        exact_df = exact_df.dropna(subset=["P/E"])
        if max_pe > 0:
            exact_df = exact_df[exact_df["P/E"] <= max_pe]
        exact_df = exact_df.sort_values(["P/E", "Company"]).drop(columns=["_has_pe"], errors="ignore")
        st.caption(
            "Separate table of every positive-P/E company passing the exact earnings rule. "
            "It never reduces the main top-10 table."
        )
        if exact_df.empty:
            st.info("No exact matches were identified from the comparable EPS data returned in this run.")
        else:
            st.dataframe(exact_df, hide_index=True, use_container_width=True)
            st.download_button(
                "Download all exact earnings matches CSV",
                exact_df.to_csv(index=False).encode("utf-8"),
                file_name=f"ftse_exact_earnings_matches_{datetime.now():%Y-%m-%d}.csv",
                mime="text/csv",
            )


# ----------------------------
# MOST DISCUSSED
# ----------------------------

with tabs[2]:
    st.subheader(
        f"Most talked about — last {discussion_days} days"
    )

    if st.button(
        "Run most-talked-about report",
        key="most_button",
    ):
        activity_df = run_activity_scan()

        top = (
            activity_df
            .dropna(subset=["Articles"])
            .sort_values(
                ["Articles", "Company"],
                ascending=[False, True]
            )
            .head(25)
        )

        st.dataframe(
            top,
            hide_index=True,
            use_container_width=True,
        )


# ----------------------------
# LEAST DISCUSSED
# ----------------------------

with tabs[3]:
    st.subheader(
        f"Least talked about — last {discussion_days} days"
    )

    if st.button(
        "Run least-talked-about report",
        key="least_button",
    ):
        activity_df = run_activity_scan()

        bottom = (
            activity_df
            .dropna(subset=["Articles"])
            .sort_values(
                ["Articles", "Company"],
                ascending=[True, True]
            )
            .head(25)
        )

        st.dataframe(
            bottom,
            hide_index=True,
            use_container_width=True,
        )


# ----------------------------
# METHOD
# ----------------------------

with tabs[4]:
    st.markdown(
        """
### Earnings rule

The app compares **four like-for-like EPS observations**. It prefers genuine **Quarterly EPS** when four comparable quarterly observations are available. If reported earnings dates indicate a roughly six-month cycle, the observations are labelled **Half-year**. If neither provides four comparable observations, **Annual EPS** can be used.

Quarterly, half-year and annual figures are **never mixed in the same four-period test**.

A company passes when:

`Latest EPS > Previous EPS > Period -2 EPS`

and:

`Period -2 EPS is NOT greater than Period -3 EPS`

The **Earnings frequency** column shows the frequency actually used.

### 7-day / 30-day setting

The 7-day or 30-day setting applies **only to online discussion activity**.

It does not affect the earnings calculation.

### P/E

The app first tries Yahoo's published trailing P/E.

If that is unavailable, it attempts an approximate P/E using:

`Yahoo valuation P/E`, `Yahoo trailing P/E`, `price / trailing EPS`, or `market capitalisation / true TTM net income`

The table tells you which method was used.

### Main top-10 and exact-match behaviour

The main value table always returns the requested number of lowest positive-P/E eligible operating companies (10 by default). Earnings status is shown alongside each company and cannot reduce the main table.

A separate table shows every positive-P/E company that passes the exact two-period earnings rule. Diagnostics show where companies are being lost because of missing P/E or comparable EPS history.

### Discussion activity

Discussion counts are free estimates based on Google News RSS results during the selected 7-day or 30-day window.

They should be used as a relative comparison rather than as an audited count of every article published online.
"""
    )
