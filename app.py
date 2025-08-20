import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Breakout AI", layout="centered")
st.title("🚀 Breakout AI")
st.caption("Type any ticker. Live data via yfinance. Shows a 0–100 Breakout Score with full factor breakdown.")

# ----------------------------
# Helpers: indicators & scoring
# ----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = pd.Series(series).dropna()
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    series = pd.Series(series).dropna()
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def zscore(value: float, lo: float, hi: float) -> float:
    if value is None or np.isnan(value):
        return 0.0
    if hi == lo:
        return 0.0
    x = (value - lo) / (hi - lo)
    return float(max(0.0, min(1.0, x)))

def pct_change(series: pd.Series, n: int) -> float:
    series = pd.Series(series).dropna()
    if len(series) <= n:
        return float("nan")
    return float((series.iloc[-1] / series.iloc[-n-1] - 1.0) * 100.0)

def last_float(x) -> float:
    """Return last scalar float from a Series/array/scalar safely."""
    if isinstance(x, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        if len(x) == 0:
            return float("nan")
        v = x[-1]
        try:
            return float(v)
        except Exception:
            return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")

# ----------------------------
# Data fetch (cached)
# ----------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_price_data(ticker: str):
    data = yf.download(ticker, period="10y", interval="1d", auto_adjust=True, progress=False, threads=False)
    if data is None or data.empty:
        raise ValueError(f"No price data for '{ticker}'.")
    close = pd.Series(data["Close"]).dropna()
    vol = pd.Series(data["Volume"]).dropna()
    close.index = pd.to_datetime(close.index)
    vol.index = pd.to_datetime(vol.index)
    return close, vol

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fundamentals(ticker: str):
    tk = yf.Ticker(ticker)
    q_earn = tk.quarterly_earnings  # columns: Revenue, Earnings (may be empty)
    q_fin = tk.quarterly_financials  # rows like Total Revenue, Gross Profit (may be empty)
    # Force DataFrames (or empty)
    if not isinstance(q_earn, pd.DataFrame):
        q_earn = pd.DataFrame()
    if not isinstance(q_fin, pd.DataFrame):
        q_fin = pd.DataFrame()
    return q_earn, q_fin

def compute_features(ticker: str):
    close, vol = fetch_price_data(ticker)

    # --- Technicals (reduce to scalars explicitly via .iloc[-1] / last_float)
    last_price = last_float(close)
    sma200_series = close.rolling(200).mean() if len(close) >= 5 else pd.Series([last_price], index=close.index)
    sma200 = last_float(sma200_series)

    rsi_series = rsi(close, 14)
    rsi14 = last_float(rsi_series)

    macd_line, signal_line, _ = macd(close)
    macd_bull = bool(last_float(macd_line) > last_float(signal_line))

    vol30 = vol.rolling(30).mean()
    vol_spike = float(last_float(vol) / last_float(vol30)) if not np.isnan(last_float(vol30)) and last_float(vol30) > 0 else 1.0

    mom_1m = pct_change(close, 21)
    mom_3m = pct_change(close, 63)

    # --- Fundamentals (best-effort from yfinance; guard every step)
    q_earn, q_fin = fetch_fundamentals(ticker)

    earnings_mom = float("nan")
    rev_accel = float("nan")
    gross_margin_exp = float("nan")

    # Earnings momentum / revenue growth (YoY proxy)
    try:
        if not q_earn.empty and "Revenue" in q_earn.columns:
            q = q_earn.tail(6).copy()
            rev = pd.to_numeric(q["Revenue"], errors="coerce").dropna()
            if len(rev) >= 5:
                earnings_mom = float((rev.iloc[-1] / rev.iloc[-5] - 1.0) * 100.0)
            elif len(rev) >= 2:
                earnings_mom = float((rev.iloc[-1] / rev.iloc[-2] - 1.0) * 100.0)
            # Acceleration: recent QoQ minus prior QoQ
            if len(rev) >= 3:
                g1 = (rev.iloc[-1] / rev.iloc[-2] - 1.0) * 100.0
                g2 = (rev.iloc[-2] / rev.iloc[-3] - 1.0) * 100.0
                rev_accel = float(g1 - g2)
    except Exception:
        pass

    # Gross margin expansion (pp change latest vs prior)
    try:
        if not q_fin.empty and "Total Revenue" in q_fin.index and "Gross Profit" in q_fin.index:
            rev_row = pd.to_numeric(q_fin.loc["Total Revenue"], errors="coerce").dropna()
            gp_row = pd.to_numeric(q_fin.loc["Gross Profit"], errors="coerce").dropna()
            common_cols = rev_row.index.intersection(gp_row.index)
            rev_row = rev_row[common_cols]
            gp_row = gp_row[common_cols]
            gm = (gp_row / rev_row * 100.0).dropna()
            # many yfinance frames are reverse-chronological; use latest two points explicitly
            if len(gm) >= 2:
                gross_margin_exp = float(gm.iloc[0] - gm.iloc[1])
    except Exception:
        pass

    feats = dict(
        price=last_price,
        sma200=sma200,
        rsi14=rsi14,
        macd_bull=macd_bull,
        vol_spike=vol_spike,
        mom_1m=mom_1m,
        mom_3m=mom_3m,
        earnings_mom=earnings_mom,
        rev_accel=rev_accel,
        gross_margin_exp=gross_margin_exp,
    )
    return feats

# ----------------------------
# Scoring weights
# ----------------------------
W_FUND = {
    "earnings_mom": 20,      # YoY revenue momentum proxy
    "rev_accel": 15,         # acceleration
    "gross_margin_exp": 15,  # margin expansion
}
W_TECH = {
    "sma200_breakout": 15,   # above 200-day
    "rsi_zone": 10,          # 50–70 ideal
    "macd_bull": 10,         # MACD line > signal
    "vol_spike": 10,         # >1.3x 30-day avg
    "momentum_combo": 5,     # 1m & 3m > 0
}

def score_fundamentals(f):
    rows = []

    # Earnings momentum (YoY revenue change %)
    em = f.get("earnings_mom", float("nan"))
    if np.isnan(em): s = 0; note="Missing/insufficient revenue history"
    else:
        s = int(round(zscore(em, -10, 40) * 100))  # -10%..+40% → 0..100
        note = f"YoY revenue Δ={em:.1f}% (−10…+40 → 0…100)"
    rows.append(("Earnings Momentum (YoY revenue)", em if not np.isnan(em) else "N/A", s, W_FUND["earnings_mom"], note))

    # Revenue acceleration
    ra = f.get("rev_accel", float("nan"))
    if np.isnan(ra): s = 0; note="Need ≥3 quarters to compute"
    else:
        s = int(round(zscore(ra, -10, 20) * 100))  # −10..+20 → 0..100
        note = f"Accel QoQ={ra:.1f} pp (−10…+20 → 0…100)"
    rows.append(("Revenue Acceleration", ra if not np.isnan(ra) else "N/A", s, W_FUND["rev_accel"], note))

    # Gross margin expansion
    gm = f.get("gross_margin_exp", float("nan"))
    if np.isnan(gm): s = 0; note="Gross margin history unavailable"
    else:
        s = int(round(zscore(gm, -5, 10) * 100))  # −5..+10 pp → 0..100
        note = f"GM Δ={gm:.1f} pp (−5…+10 → 0…100)"
    rows.append(("Gross Margin Expansion", gm if not np.isnan(gm) else "N/A", s, W_FUND["gross_margin_exp"], note))

    return rows

def score_technicals(f):
    rows = []

    # 200‑DMA breakout (use scalar %)
    if f.get("sma200", 0) and not np.isnan(f["sma200"]):
        pct = (f["price"] / f["sma200"] - 1.0) * 100.0
    else:
        pct = float("nan")
    s_dma = int(round(zscore(pct, -5, 10) * 100)) if not np.isnan(pct) else 0
    val_dma = f"{pct:.1f}% vs 200‑DMA" if not np.isnan(pct) else "N/A"
    rows.append(("200‑Day Breakout", val_dma, s_dma, W_TECH["sma200_breakout"], "Higher above (up to ~+10%) scores better"))

    # RSI zone
    r = f.get("rsi14", float("nan"))
    s_rsi = int(round(zscore(r, 40, 70) * 100)) if not np.isnan(r) else 0
    val_rsi = f"{r:.1f}" if not np.isnan(r) else "N/A"
    rows.append(("RSI Zone (14)", val_rsi, s_rsi, W_TECH["rsi_zone"], "50–70 preferred; <40 weak, >80 overbought"))

    # MACD bull (bool → 100/0)
    mb = bool(f.get("macd_bull", False))
    s_macd = 100 if mb else 0
    rows.append(("MACD Bullish Cross", "Yes" if mb else "No", s_macd, W_TECH["macd_bull"], "MACD line > signal"))

    # Volume spike vs 30‑day avg
    vs = f.get("vol_spike", float("nan"))
    s_vol = int(round(zscore(vs, 1.0, 2.0) * 100)) if not np.isnan(vs) else 0
    val_vs = f"{vs:.2f}×" if not np.isnan(vs) else "N/A"
    rows.append(("Volume Spike (x30d avg)", val_vs, s_vol, W_TECH["vol_spike"], "Breakouts with >1.3× volume are stronger"))

    # Momentum combo (1m & 3m > 0)
    m1 = f.get("mom_1m", float("nan"))
    m3 = f.get("mom_3m", float("nan"))
    combo = (not np.isnan(m1) and m1 > 0) and (not np.isnan(m3) and m3 > 0)
    s_mom = 100 if combo else 0
    val_mom = f"1m={m1:.1f}%, 3m={m3:.1f}%" if (not np.isnan(m1) and not np.isnan(m3)) else "N/A"
    rows.append(("Momentum (1m & 3m)", val_mom, s_mom, W_TECH["momentum_combo"], "Both > 0% preferred"))

    return rows

def weighted_score(rows):
    total_w = sum(w for _,_,_,w,_ in rows)
    if total_w <= 0:
        return 0
    return int(round(sum(s * w for _,_,s,w,_ in rows) / total_w))

# ----------------------------
# UI
# ----------------------------
ticker = st.text_input("Ticker", value="ASTS").strip().upper()
if st.button("Run Screener"):
    try:
        with st.spinner("Fetching data & computing signals…"):
            feats = compute_features(ticker)

            fund_rows = score_fundamentals(feats)
            tech_rows = score_technicals(feats)

            fund_score = weighted_score(fund_rows)
            tech_score = weighted_score(tech_rows)
            final_score = int(round((fund_score * 0.50) + (tech_score * 0.50)))

        st.subheader(f"Result for {ticker}")
        st.metric("Breakout Score (0–100)", final_score)
        if final_score >= 75:
            st.success("🔥 Strong Breakout Setup")
        elif final_score >= 55:
            st.info("🟡 Constructive / Watchlist")
        else:
            st.warning("🔴 Weak / Avoid")

        # Fundamentals table
        st.markdown("### Fundamentals Breakdown")
        fund_df = pd.DataFrame([{
            "Factor": n, "Value": v, "Subscore (0‑100)": s, "Weight": w, "Notes": note
        } for (n,v,s,w,note) in fund_rows])
        st.dataframe(fund_df, use_container_width=True)

        # Technicals table
        st.markdown("### Technicals Breakdown")
        tech_df = pd.DataFrame([{
            "Factor": n, "Value": v, "Subscore (0‑100)": s, "Weight": w, "Notes": note
        } for (n,v,s,w,note) in tech_rows])
        st.dataframe(tech_df, use_container_width=True)

        with st.expander("Debug / raw features"):
            st.json(feats)

        st.caption("Notes: Earnings/financial fields are best‑effort from yfinance. We can add premium feeds later (DCF, insiders, short interest, options) and they’ll appear in these tables with weights.")

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        st.caption("If this persists, try another ticker or retry in a minute.")