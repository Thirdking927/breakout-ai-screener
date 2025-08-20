import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ---------------------------
# Utilities: fetch + features
# ---------------------------

@st.cache_data(show_spinner=False)
def fetch_history_yahoo_csv(ticker: str, years: int = 10) -> pd.DataFrame:
    """
    Pull daily adjusted prices from Yahoo Finance via their CSV endpoint.
    No extra packages needed (works on Streamlit Cloud).
    """
    end = int(datetime.now(timezone.utc).timestamp())
    start = int((datetime.now(timezone.utc) - timedelta(days=365*years)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        f"?period1={start}&period2={end}&interval=1d&events=history&includeAdjustedClose=true"
    )
    df = pd.read_csv(url)
    # Use Adjusted Close for signals
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.rename(columns={"Adj Close": "AdjClose"})
    return df[["Date", "AdjClose"]].dropna()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def pct_change_n(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return float("nan")
    return float((series.iloc[-1] / series.iloc[-days-1] - 1.0) * 100.0)

def weekly_ma(series: pd.Series, window_weeks: int = 200) -> float:
    wk = series.resample("W-FRI").last().dropna()
    if len(wk) >= window_weeks:
        return float(wk.rolling(window_weeks).mean().iloc[-1])
    # Fallback if not enough history
    return float(wk.mean())

def build_features(ticker: str) -> dict:
    df = fetch_history_yahoo_csv(ticker)
    close = df.set_index("Date")["AdjClose"]
    last_price = float(close.iloc[-1])

    rsi14 = float(rsi(close, 14).iloc[-1])
    mom_1m = pct_change_n(close, 21)   # ~21 trading days
    mom_3m = pct_change_n(close, 63)   # ~63 trading days
    wma200 = weekly_ma(close, 200)

    return {
        "price": last_price,
        "rsi14": rsi14,
        "mom_1m_pct": mom_1m,
        "mom_3m_pct": mom_3m,
        "wma200": wma200,
    }

# ---------------------------
# Scoring (self-contained)
# ---------------------------

def _z(x, lo, hi):
    if hi == lo: 
        return 0.0
    s = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, s)))

def score_from_features(feats: dict) -> dict:
    price = feats["price"]
    wma200 = feats["wma200"]
    rsi14 = feats["rsi14"]
    m1 = feats["mom_1m_pct"]
    m3 = feats["mom_3m_pct"]

    # Distance to 200WMA: reward from -20% .. +10% (peak near slightly above)
    pct_to_wma = (price - wma200) / wma200 * 100.0 if wma200 else 0.0
    near_wma = _z(pct_to_wma, -20.0, 10.0)

    # RSI sweet spot 40..70
    rsi_sweet = _z(rsi14, 40.0, 70.0)

    # Momentum windows
    mom1 = _z(m1, -10.0, 20.0)
    mom3 = _z(m3, -15.0, 35.0)

    # Weights sum to 1
    w = {"near_wma": 0.35, "rsi_sweet": 0.20, "mom1": 0.20, "mom3": 0.25}
    raw = near_wma * w["near_wma"] + rsi_sweet * w["rsi_sweet"] + mom1 * w["mom1"] + mom3 * w["mom3"]
    score = int(round(raw * 100))

    if score >= 75:
        band, color = "Strong Buy", "green"
    elif score >= 55:
        band, color = "Constructive / Watchlist", "yellow"
    else:
        band, color = "Weak / Avoid", "red"

    return {
        "score_0_100": score,
        "band": band,
        "color": color,
        "contributions": {
            "near_200WMA": int(round(near_wma * 100)),
            "RSI_sweet": int(round(rsi_sweet * 100)),
            "Momentum_1m": int(round(mom1 * 100)),
            "Momentum_3m": int(round(mom3 * 100)),
        },
    }

# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Breakout AI Screener", layout="centered")
st.title("🚀 Breakout AI Screener")
st.caption("Single‑file app. Type a symbol, fetches data, and outputs a 0–100 buy score with a color band.")

ticker = st.text_input("Ticker", value="ASTS").upper()

if st.button("Run Screener"):
    try:
        with st.spinner("Fetching data…"):
            feats = build_features(ticker)
        res = score_from_features(feats)

        st.subheader(f"Result for {ticker}")
        st.metric("Buy Score (0–100)", res["score_0_100"])
        st.write(f"**Signal:** {res['band']}")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Auto‑fetched inputs**")
            st.json(feats)
        with col2:
            st.write("**Per‑factor (0–100)**")
            st.json(res["contributions"])

        st.info("Tip: On iPhone → Safari → Share → **Add to Home Screen** for a full‑screen app icon.")

    except Exception as e:
        st.error(f"Error: {e}")