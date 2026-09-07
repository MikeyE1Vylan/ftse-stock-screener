
# FTSE Value + Stock Attention Screener

This Streamlit app combines two screens across the FTSE 100 and FTSE 250.

## 1. Low P/E + exactly two earnings increases

A share qualifies where the latest four quarterly earnings observations, latest first, satisfy:

`Q0 > Q1 > Q2` and **NOT** `Q2 > Q3`

The app then ranks qualifying stocks from lowest positive trailing P/E to highest.

## 2. Online discussion / article activity

The app estimates how much each FTSE company has been discussed online using Google News RSS search results.

You can select:
- last 7 days
- last 30 days

You can run the reports on any day.

The app reports:
- estimated article count
- activity rank across the chosen FTSE universe
- activity percentile
- most-discussed stocks
- least-discussed stocks
- sample headlines

## Combined report

The combined report links the two screens. Every stock that qualifies for the P/E + earnings rule
is matched to its article count, discussion rank, percentile, and discussion band.

This makes it easy to identify shares that are:
- cheap and already heavily discussed
- cheap with improving earnings but still relatively unnoticed
- or anywhere in between

## Free-data limitation

Google News RSS does not provide a guaranteed exhaustive count of every article published online.
The activity figures are therefore best used as **relative estimates** for comparing companies within
the same screen and time window.

Yahoo Finance data accessed through `yfinance` can also contain gaps or delays.

## Run on Windows

1. Install Python 3.11 or newer.
2. Open Command Prompt in this folder.
3. Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or double-click `RUN_APP.bat`.

## Weekly reports

`monday_combined_report.py` produces a combined 7-day report every time it is run.
You can schedule it in Windows Task Scheduler for Monday mornings.

The browser app itself can be run on any day using either the 7-day or 30-day window.
