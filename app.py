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
st.caption(
    "FTSE 100 + FTSE 250 • lowest P/E • latest reported earnings trend • "
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

        out = table[[company_col, ticker_col]].copy()
        out.columns = ["Company", "Ticker"]
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


def get_latest_four_earnings(ticker_obj):
    """
    Collect the four most recent available earnings observations.

    Priority:
      1. Reported EPS from earnings dates
      2. Quarterly financial-statement EPS
      3. Annual financial-statement EPS

    The dates can be any age. There is NO 7-day or 30-day restriction here.
    """
    candidates = []

    candidates.extend(_collect_reported_eps_from_earnings_dates(ticker_obj))

    try:
        candidates.extend(
            _collect_eps_from_statement(
                ticker_obj.quarterly_financials,
                "Quarterly EPS"
            )
        )
    except Exception:
        pass

    try:
        candidates.extend(
            _collect_eps_from_statement(
                ticker_obj.financials,
                "Annual EPS"
            )
        )
    except Exception:
        pass

    if not candidates:
        return []

    # Sort newest first and deduplicate near-identical reporting dates.
    candidates = sorted(
        candidates,
        key=lambda x: x["date"],
        reverse=True
    )

    deduped = []

    for item in candidates:
        keep = True

        for existing in deduped:
            # Treat dates within 14 days as the same reporting period.
            if abs((item["date"] - existing["date"]).days) <= 14:
                keep = False
                break

        if keep:
            deduped.append(item)

        if len(deduped) >= 4:
            break

    return deduped[:4]


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


def calculate_pe(ticker_obj):
    """
    Return (pe, method).

    Preferred:
      Yahoo trailing P/E

    Fallback:
      market cap / latest annual net income

    This avoids relying entirely on Yahoo's info endpoint.
    """
    info = _safe_info(ticker_obj)

    pe = info.get("trailingPE")
    if isinstance(pe, (int, float)) and math.isfinite(pe) and pe > 0:
        return float(pe), "Yahoo trailing P/E"

    fast = _safe_fast_info(ticker_obj)

    market_cap = None
    try:
        market_cap = fast.get("market_cap")
    except Exception:
        try:
            market_cap = fast["market_cap"]
        except Exception:
            pass

    if not market_cap:
        market_cap = info.get("marketCap")

    if market_cap:
        try:
            annual = ticker_obj.financials
            net_income = _latest_positive_net_income(annual)

            if net_income and net_income > 0:
                approx_pe = float(market_cap) / float(net_income)

                if math.isfinite(approx_pe) and approx_pe > 0:
                    return approx_pe, "Market cap / latest annual net income"
        except Exception:
            pass

    return None, None


# ============================================================
# STOCK FUNDAMENTAL ROW
# ============================================================

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_stock_fundamentals(ticker, company, index_name):
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
            "Earnings rule": status,
            "Report 1 date": dates[0],
            "Report 1 EPS": values[0],
            "Report 2 date": dates[1],
            "Report 2 EPS": values[1],
            "Report 3 date": dates[2],
            "Report 3 EPS": values[2],
            "Report 4 date": dates[3],
            "Report 4 EPS": values[3],
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

    show_pass_only = st.checkbox(
        "Show only exact earnings matches",
        value=False,
    )

    st.markdown(
        """
**Important**

The **7 / 30 day setting applies only to news discussion**.

The earnings test always uses the **four most recent available reported earnings periods**, regardless of how old they are.
"""
    )


constituents = all_constituents[
    all_constituents["Index"].isin(index_filter)
].reset_index(drop=True)


# ============================================================
# RUNNERS
# ============================================================

def run_fundamental_scan():
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

    if show_pass_only:
        pe_available = pe_available[
            pe_available["Earnings rule"]
            == "PASS — exactly last 2 increased"
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
        "exact_matches": len(
            raw[
                raw["Earnings rule"]
                == "PASS — exactly last 2 increased"
            ]
        ),
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
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "FTSE shares checked",
        diag["constituents"]
    )

    c2.metric(
        "P/E available",
        diag["pe_available"]
    )

    c3.metric(
        "Exact earnings matches",
        diag["exact_matches"]
    )

    c4.metric(
        "Rows returned",
        diag["returned"]
    )


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
        value_df, diag, _ = run_fundamental_scan()

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
        value_df, diag, raw = run_fundamental_scan()

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

For each company, the app looks for the **four most recent available reported earnings observations**.

There is **no age restriction** on these reports.

A company passes when:

`Report 1 EPS > Report 2 EPS > Report 3 EPS`

and:

`Report 3 EPS is NOT greater than Report 4 EPS`

This means earnings improved across the latest two reporting transitions, but the company was not already on a three-or-more-period run.

### 7-day / 30-day setting

The 7-day or 30-day setting applies **only to online discussion activity**.

It does not affect the earnings calculation.

### P/E

The app first tries Yahoo's published trailing P/E.

If that is unavailable, it attempts an approximate P/E using:

`market capitalisation / latest annual net income`

The table tells you which method was used.

### Always-return behaviour

Unless **Show only exact earnings matches** is ticked, the value report does not require a company to pass the earnings rule.

It ranks all shares with a usable positive P/E from lowest to highest and displays their earnings status alongside them.

Therefore the normal report should return a top 10 even when fewer than ten companies satisfy the exact earnings condition.

### Discussion activity

Discussion counts are free estimates based on Google News RSS results during the selected 7-day or 30-day window.

They should be used as a relative comparison rather than as an audited count of every article published online.
"""
    )
