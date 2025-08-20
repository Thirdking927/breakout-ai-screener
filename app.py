import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Breakout AI", layout="centered")
st.title("🚀 Breakout AI")
st.caption("Any ticker. Live data via yfinance. Breakout Score (0–100) with full factor breakdown, calibrated by the stock’s own history + market regime.")

# ========== Helpers ==========
def to_series_1d(x, index=None) -> pd.Series:
    if isinstance(x, pd.Series): return x
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0] if x.shape[1] >= 1 else pd.Series(dtype=float)
    arr = np.asarray(x).reshape(-1)
    return pd.Series(arr, index=index) if index is not None else pd.Series(arr)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = to_series_1d(series).dropna()
    d = s.diff()
    gain = (d.where(d > 0, 0)).rolling(period).mean()
    loss = (-d.where(d < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    s = to_series_1d(series).dropna()
    ema_f = s.ewm(span=fast, adjust=False).mean()
    ema_s = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def pct_change(series: pd.Series, n: int) -> float:
    s = to_series_1d(series).dropna()
    if len(s) <= n: return float("nan")
    return float((s.iloc[-1] / s.iloc[-n-1] - 1.0) * 100.0)

def last_float(x) -> float:
    try:
        if isinstance(x, (pd.Series, pd.Index)): 
            return float(x.iloc[-1]) if len(x) else float("nan")
        if isinstance(x, pd.DataFrame): 
            return float(x.iloc[-1, 0]) if len(x) else float("nan")
        arr = np.asarray(x).reshape(-1)
        return float(arr[-1])
    except Exception:
        return float("nan")

def percentile_score(current: float, hist: pd.Series) -> int:
    """Percentile (0-100) of current vs. historical series (dropna)."""
    h = to_series_1d(hist).dropna()
    if not np.isfinite(current) or h.empty: return 0
    return int(round(100 * (h <= current).mean()))

# ========== Data fetch (cached) ==========
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_prices(ticker: str, years: int = 10):
    df = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError(f"No price data for '{ticker}'.")
    close = to_series_1d(df["Close"], index=df.index).dropna()
    vol = to_series_1d(df["Volume"], index=df.index).dropna()
    close.index = pd.to_datetime(close.index); vol.index = pd.to_datetime(vol.index)
    return close, vol

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fundamentals(ticker: str):
    tk = yf.Ticker(ticker)
    q_earn = tk.quarterly_earnings if isinstance(tk.quarterly_earnings, pd.DataFrame) else pd.DataFrame()
    q_fin  = tk.quarterly_financials if isinstance(tk.quarterly_financials, pd.DataFrame) else pd.DataFrame()
    return q_earn, q_fin

def compute_features(ticker: str):
    close, vol = fetch_prices(ticker)
    # history window for percentile calibration
    N = min(len(close), 252)  # ~1y trading days
    c_hist = close.tail(N)
    v_hist = vol.tail(N)

    # Technicals (raw)
    price = last_float(close)
    sma200_series = close.rolling(200).mean() if len(close) >= 200 else close.rolling(min(len(close), 50)).mean()
    sma200 = last_float(sma200_series)
    pct_to_200 = (price / sma200 - 1.0) * 100.0 if np.isfinite(price) and np.isfinite(sma200) and sma200>0 else float("nan")

    rsi_series = rsi(close, 14)
    rsi14 = last_float(rsi_series)

    macd_line, sig_line, _ = macd(close)
    macd_bull = bool(last_float(macd_line) > last_float(sig_line))

    vol30 = vol.rolling(30).mean()
    vol_spike = float(last_float(vol) / last_float(vol30)) if np.isfinite(last_float(vol30)) and last_float(vol30) > 0 else 1.0

    mom_1m = pct_change(close, 21)
    mom_3m = pct_change(close, 63)

    # Hist series for percentiles
    pct_to_200_hist = ((c_hist / (c_hist.rolling(200).mean())) - 1.0) * 100.0 if len(c_hist) >= 200 else ((c_hist / (c_hist.rolling(50).mean())) - 1.0) * 100.0
    rsi_hist = rsi_series.tail(N)
    volspike_hist = (v_hist / v_hist.rolling(30).mean())
    mom1_hist = c_hist.pct_change(21) * 100.0
    mom3_hist = c_hist.pct_change(63) * 100.0

    # Fundamentals (best-effort)
    q_earn, q_fin = fetch_fundamentals(ticker)
    earnings_mom = float("nan"); rev_accel = float("nan"); gm_exp = float("nan")
    try:
        if not q_earn.empty and "Revenue" in q_earn.columns:
            q = q_earn.tail(6).copy()
            rev = pd.to_numeric(q["Revenue"], errors="coerce").dropna()
            if len(rev) >= 5:
                earnings_mom = float((rev.iloc[-1] / rev.iloc[-5] - 1.0) * 100.0)
            elif len(rev) >= 2:
                earnings_mom = float((rev.iloc[-1] / rev.iloc[-2] - 1.0) * 100.0)
            if len(rev) >= 3:
                g1 = (rev.iloc[-1] / rev.iloc[-2] - 1.0) * 100.0
                g2 = (rev.iloc[-2] / rev.iloc[-3] - 1.0) * 100.0
                rev_accel = float(g1 - g2)
    except Exception: pass
    try:
        if not q_fin.empty and "Total Revenue" in q_fin.index and "Gross Profit" in q_fin.index:
            rev_row = pd.to_numeric(q_fin.loc["Total Revenue"], errors="coerce").dropna()
            gp_row  = pd.to_numeric(q_fin.loc["Gross Profit"], errors="coerce").dropna()
            common  = rev_row.index.intersection(gp_row.index)
            gm = (gp_row[common] / rev_row[common] * 100.0).dropna()
            if len(gm) >= 2:
                gm_exp = float(gm.iloc[0] - gm.iloc[1])
    except Exception: pass

    # Relative strength vs SPY (3m)
    spy_close, _ = fetch_prices("SPY")
    rs_3m = float("nan")
    try:
        if len(close) > 63 and len(spy_close) > 63:
            rs_3m = pct_change(close, 63) - pct_change(spy_close.reindex(close.index).ffill(), 63)
    except Exception: pass

    feats = dict(
        price=price, sma200=sma200, pct_to_200=pct_to_200,
        rsi14=rsi14, macd_bull=macd_bull, vol_spike=vol_spike,
        mom_1m=mom_1m, mom_3m=mom_3m,
        pct_to_200_hist=pct_to_200_hist, rsi_hist=rsi_hist,
        volspike_hist=volspike_hist, mom1_hist=mom1_hist, mom3_hist=mom3_hist,
        earnings_mom=earnings_mom, rev_accel=rev_accel, gross_margin_exp=gm_exp,
        rs_3m=rs_3m
    )
    return feats

# Market regime (+5 if SPY > 200DMA, +3 if VIX<20)
@st.cache_data(show_spinner=False, ttl=900)
def market_regime_bonus():
    try:
        spy, _ = fetch_prices("SPY")
        spy_price = last_float(spy)
        spy_200 = last_float(spy.rolling(200).mean())
        bull = (spy_price > spy_200) if np.isfinite(spy_price) and np.isfinite(spy_200) else False
    except Exception:
        bull = False
    vix = yf.download("^VIX", period="1y", interval="1d", progress=False, threads=False)
    vix_last = float(vix["Close"].iloc[-1]) if isinstance(vix, pd.DataFrame) and not vix.empty else float("nan")
    calm = (vix_last < 20) if np.isfinite(vix_last) else False
    bonus = (5 if bull else 0) + (3 if calm else 0)
    return bonus, {"SPY>200DMA": bull, "VIX<20": calm}

# ========== Weights: Tech 80 / Fund 20 ==========
W_FUND = {"earnings_mom": 10, "rev_accel": 5, "gross_margin_exp": 5}  # total 20
W_TECH = {
    "rel_strength_3m": 20,
    "pct_to_200_pctile": 20,
    "vol_spike_pctile": 15,
    "rsi_pctile": 15,
    "momentum_pctile": 10
}  # subtotal 80 (plus MACD flag included inside momentum_pctile calc commentary or add as note)

def score_block(ticker_feats):
    f = ticker_feats
    rows_f, rows_t = [], []

    # ---- Technicals via percentiles (vs own 1y history) ----
    s_pct200 = percentile_score(f["pct_to_200"], f["pct_to_200_hist"])
    rows_t.append(("Pct vs 200‑DMA (percentile)", f"{f['pct_to_200']:.1f}%", s_pct200, W_TECH["pct_to_200_pctile"], "Higher vs own 1y baseline"))

    s_rsi = percentile_score(f["rsi14"], f["rsi_hist"])
    rows_t.append(("RSI(14) (percentile)", f"{f['rsi14']:.1f}", s_rsi, W_TECH["rsi_pctile"], "RSI relative to own history"))

    s_vol = percentile_score(f["vol_spike"], f["volspike_hist"])
    rows_t.append(("Volume/30d (percentile)", f"{f['vol_spike']:.2f}×", s_vol, W_TECH["vol_spike_pctile"], "Volume expansion vs own history"))

    s_m1 = percentile_score(f["mom_1m"], f["mom1_hist"])
    s_m3 = percentile_score(f["mom_3m"], f["mom3_hist"])
    s_mom = int(round((s_m1 + s_m3) / 2))
    rows_t.append(("Momentum (1m & 3m percentiles)", f"1m={f['mom_1m']:.1f}%, 3m={f['mom_3m']:.1f}%", s_mom, W_TECH["momentum_pctile"], "Averaged percentiles"))

    # Relative strength vs SPY (3m)
    rs_val = f["rs_3m"]
    # map RS (−20…+20%) to 0–100 using simple min‑max ▸ still percentile‑ish
    def minmax(v, lo=-20, hi=20):
        if not np.isfinite(v): return 0
        x = (v - lo) / (hi - lo)
        return int(round(100 * max(0, min(1, x))))
    s_rs = minmax(rs_val, -20, 20)
    rows_t.append(("Relative Strength vs SPY (3m)", f"{rs_val:.1f}%", s_rs, W_TECH["rel_strength_3m"], "Outperform market last 3m"))

    # ---- Fundamentals (lighter) ----
    em = f.get("earnings_mom", float("nan"))
    s_em = minmax(em, -20, 50)
    rows_f.append(("Earnings Momentum (YoY revenue)", em if np.isfinite(em) else "N/A", s_em, W_FUND["earnings_mom"], "−20…+50% mapped"))

    ra = f.get("rev_accel", float("nan"))
    s_ra = minmax(ra, -15, 25)
    rows_f.append(("Revenue Acceleration (pp)", ra if np.isfinite(ra) else "N/A", s_ra, W_FUND["rev_accel"], "−15…+25 pp mapped"))

    gm = f.get("gross_margin_exp", float("nan"))
    s_gm = minmax(gm, -8, 15)
    rows_f.append(("Gross Margin Expansion (pp)", gm if np.isfinite(gm) else "N/A", s_gm, W_FUND["gross_margin_exp"], "−8…+15 pp mapped"))

    return rows_f, rows_t

def weighted_score(rows):
    tw = sum(w for *_ , w, _ in rows)
    if tw <= 0: return 0
    return int(round(sum(s * w for *_, s, w, _ in rows) / tw))

# ========== UI ==========
ticker = st.text_input("Ticker", value="ASTS").strip().upper()
if st.button("Run Screener"):
    try:
        with st.spinner("Fetching & scoring…"):
            feats = compute_features(ticker)
            fund_rows, tech_rows = score_block(feats)
            raw_score = weighted_score(fund_rows + tech_rows)

            bonus, regime_flags = market_regime_bonus()
            final_score = int(min(100, max(0, raw_score + bonus)))

        st.subheader(f"Result for {ticker}")
        st.metric("Breakout Score (0–100)", final_score)
        badge = "🔥 Strong Breakout Setup" if final_score >= 70 else "🟡 Constructive / Watchlist" if final_score >= 50 else "🔴 Weak / Avoid"
        (st.success if final_score >= 70 else st.info if final_score >= 50 else st.warning)(badge)

        st.markdown("### Technicals Breakdown")
        st.dataframe(pd.DataFrame([{"Factor":n,"Value":v,"Subscore (0‑100)":s,"Weight":w,"Notes":note} for (n,v,s,w,note) in tech_rows]), use_container_width=True)

        st.markdown("### Fundamentals Breakdown")
        st.dataframe(pd.DataFrame([{"Factor":n,"Value":v,"Subscore (0‑100)":s,"Weight":w,"Notes":note} for (n,v,s,w,note) in fund_rows]), use_container_width=True)

        st.markdown("### Market Regime")
        st.json({"Bonus": f"+{bonus} pts", **regime_flags})

        with st.expander("Debug / raw features"):
            st.json({k:(float(v) if isinstance(v,(int,float,np.floating)) else (None if v is None else str(v)) ) for k,v in feats.items() if k not in ["pct_to_200_hist","rsi_hist","volspike_hist","mom1_hist","mom3_hist"]})

        st.caption("Notes: Factors are percentiled vs the stock’s own last ~252 trading days and adjusted for market regime. Fundamentals lighter by design so strong technicals can surface.")

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        st.caption("If this persists, try another ticker or retry in a minute.")