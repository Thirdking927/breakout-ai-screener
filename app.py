import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Breakout AI", layout="centered")
st.title("🚀 Breakout AI")
st.caption("Any ticker. Live data via yfinance. Breakout Score (0–100) with full factor breakdown, using z-scores vs the stock’s own history + market regime + confluence bonus.")

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

def zscore_to_score(current: float, hist: pd.Series, cap_sigma: float = 2.0) -> tuple[int, float]:
    """
    Convert current value to a 0-100 score using z-score vs hist:
    score = clip(50 + 25*z, 0, 100), with z capped to +/- cap_sigma for stability.
    Returns (score, z).
    """
    h = to_series_1d(hist).dropna()
    if not np.isfinite(current) or h.empty: 
        return 0, float("nan")
    mu = float(h.mean())
    sd = float(h.std(ddof=0))
    if sd == 0:
        return 50, 0.0
    z = (current - mu) / sd
    z = max(-cap_sigma, min(cap_sigma, z))
    score = int(round(max(0.0, min(100.0, 50 + 25*z))))
    return score, z

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
    N = min(len(close), 252)  # ~1y window for calibration
    c_hist = close.tail(N)
    v_hist = vol.tail(N)

    # --- Technicals (raw)
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

    # --- Hist series for z-score scaling
    pct_to_200_hist = ((c_hist / (c_hist.rolling(200).mean())) - 1.0) * 100.0 if len(c_hist) >= 200 else ((c_hist / (c_hist.rolling(50).mean())) - 1.0) * 100.0
    rsi_hist = rsi_series.tail(N)
    volspike_hist = (v_hist / v_hist.rolling(30).mean())
    mom1_hist = c_hist.pct_change(21) * 100.0
    mom3_hist = c_hist.pct_change(63) * 100.0

    # --- Fundamentals (light) ---
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

    # --- Relative strength vs SPY (3m) + its history for z-score ---
    spy_close, _ = fetch_prices("SPY")
    rs_3m = float("nan"); rs_3m_hist = pd.Series(dtype=float)
    try:
        # Build rolling RS time series over the same index as c_hist
        spy_aligned = spy_close.reindex(close.index).ffill()
        rs_series = (close.pct_change(63) - spy_aligned.pct_change(63)) * 100.0
        rs_3m = last_float(rs_series)
        rs_3m_hist = rs_series.tail(N)
    except Exception: pass

    return dict(
        # raw
        price=price, sma200=sma200, pct_to_200=pct_to_200,
        rsi14=rsi14, macd_bull=macd_bull, vol_spike=vol_spike,
        mom_1m=mom_1m, mom_3m=mom_3m, rs_3m=rs_3m,
        # history for z
        pct_to_200_hist=pct_to_200_hist, rsi_hist=rsi_hist,
        volspike_hist=volspike_hist, mom1_hist=mom1_hist, mom3_hist=mom3_hist,
        rs_3m_hist=rs_3m_hist,
        # fundamentals
        earnings_mom=earnings_mom, rev_accel=rev_accel, gross_margin_exp=gm_exp
    )

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
    "pct_to_200_z": 20,
    "vol_spike_z": 15,
    "rsi_z": 15,
    "momentum_z": 10
}  # total 80

def score_blocks(f):
    rows_f, rows_t = [], []

    # ---- Technicals via z-score vs own 1y history ----
    s_pct200, z_pct200 = zscore_to_score(f["pct_to_200"], f["pct_to_200_hist"])
    rows_t.append(("Pct vs 200‑DMA (z)", f"{f['pct_to_200']:.1f}%", s_pct200, W_TECH["pct_to_200_z"], f"z={z_pct200:.2f}σ vs 1y"))

    s_rsi, z_rsi = zscore_to_score(f["rsi14"], f["rsi_hist"])
    rows_t.append(("RSI(14) (z)", f"{f['rsi14']:.1f}", s_rsi, W_TECH["rsi_z"], f"z={z_rsi:.2f}σ vs 1y"))

    s_vol, z_vol = zscore_to_score(f["vol_spike"], f["volspike_hist"])
    rows_t.append(("Volume/30d (z)", f"{f['vol_spike']:.2f}×", s_vol, W_TECH["vol_spike_z"], f"z={z_vol:.2f}σ vs 1y"))

    s_m1, z_m1 = zscore_to_score(f["mom_1m"], f["mom1_hist"])
    s_m3, z_m3 = zscore_to_score(f["mom_3m"], f["mom3_hist"])
    s_mom = int(round((s_m1 + s_m3) / 2))
    rows_t.append(("Momentum (1m & 3m z)", f"1m={f['mom_1m']:.1f}%, 3m={f['mom_3m']:.1f}%", s_mom, W_TECH["momentum_z"], f"z1m={z_m1:.2f}, z3m={z_m3:.2f}"))

    s_rs, z_rs = zscore_to_score(f["rs_3m"], f["rs_3m_hist"])
    rows_t.append(("Relative Strength vs SPY (3m z)", f"{f['rs_3m']:.1f}%", s_rs, W_TECH["rel_strength_3m"], f"z={z_rs:.2f}σ vs 1y"))

    # ---- Fundamentals (light, min-max style mapping kept) ----
    def map_minmax(v, lo, hi):
        if not np.isfinite(v): return 0
        x = (v - lo) / (hi - lo)
        return int(round(100 * max(0, min(1, x))))
    em = f.get("earnings_mom", float("nan"))
    s_em = map_minmax(em, -20, 50)
    rows_f.append(("Earnings Momentum (YoY revenue)", em if np.isfinite(em) else "N/A", s_em, W_FUND["earnings_mom"], "−20…+50% mapped"))

    ra = f.get("rev_accel", float("nan"))
    s_ra = map_minmax(ra, -15, 25)
    rows_f.append(("Revenue Acceleration (pp)", ra if np.isfinite(ra) else "N/A", s_ra, W_FUND["rev_accel"], "−15…+25 pp mapped"))

    gm = f.get("gross_margin_exp", float("nan"))
    s_gm = map_minmax(gm, -8, 15)
    rows_f.append(("Gross Margin Expansion (pp)", gm if np.isfinite(gm) else "N/A", s_gm, W_FUND["gross_margin_exp"], "−8…+15 pp mapped"))

    return rows_f, rows_t

def weighted_score(rows):
    tw = sum(w for *_ , w, _ in rows)
    if tw <= 0: return 0
    return int(round(sum(s * w for *_, s, w, _ in rows) / tw))

def confluence_bonus(tech_rows, threshold=70, need=3, bonus=5):
    hits = sum(1 for (_, _, s, _, _) in tech_rows if s >= threshold)
    return bonus if hits >= need else 0, hits

# ========== UI ==========
ticker = st.text_input("Ticker", value="ASTS").strip().upper()
if st.button("Run Screener"):
    try:
        with st.spinner("Fetching & scoring…"):
            feats = compute_features(ticker)
            fund_rows, tech_rows = score_blocks(feats)

            base = weighted_score(fund_rows + tech_rows)
            conf_bonus, hits = confluence_bonus(tech_rows, threshold=70, need=3, bonus=5)
            mr_bonus, regime_flags = market_regime_bonus()
            final_score = int(min(100, max(0, base + conf_bonus + mr_bonus)))

        st.subheader(f"Result for {ticker}")
        st.metric("Breakout Score (0–100)", final_score)
        badge = "🔥 Strong Breakout Setup" if final_score >= 70 else "🟡 Constructive / Watchlist" if final_score >= 50 else "🔴 Weak / Avoid"
        (st.success if final_score >= 70 else st.info if final_score >= 50 else st.warning)(badge)

        st.markdown("### Technicals Breakdown")
        st.dataframe(pd.DataFrame([{"Factor":n,"Value":v,"Subscore (0‑100)":s,"Weight":w,"Notes":note} for (n,v,s,w,note) in tech_rows]), use_container_width=True)

        st.markdown("### Fundamentals Breakdown")
        st.dataframe(pd.DataFrame([{"Factor":n,"Value":v,"Subscore (0‑100)":s,"Weight":w,"Notes":note} for (n,v,s,w,note) in fund_rows]), use_container_width=True)

        st.markdown("### Bonuses")
        st.json({
            "Confluence bonus": f"+{conf_bonus} (tech factors ≥70: {hits})",
            "Market regime bonus": f"+{mr_bonus}",
            **regime_flags
        })

        with st.expander("Debug / raw features"):
            st.json({k:(float(v) if isinstance(v,(int,float,np.floating)) else (None if v is None else str(v)) ) 
                     for k,v in feats.items() 
                     if k not in ["pct_to_200_hist","rsi_hist","volspike_hist","mom1_hist","mom3_hist","rs_3m_hist"]})

        st.caption("Technical subscores use z-scores vs the stock’s own ~1y history. Confluence and market regime bonuses help strong setups actually surface as green.")

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        st.caption("If this persists, try another ticker or retry in a minute.")