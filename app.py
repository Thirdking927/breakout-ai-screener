import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Breakout AI", layout="centered")
st.title("🚀 Breakout AI")
st.caption("Type any ticker. Live data via yfinance. 0–100 Breakout Score with full factor breakdown.")

# ========== Helpers ==========
def to_series_1d(x, index=None) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        return x.iloc[:, 0]
    arr = np.asarray(x).reshape(-1)
    return pd.Series(arr, index=index) if index is not None else pd.Series(arr)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = to_series_1d(series).dropna()
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    s = to_series_1d(series).dropna()
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def zscore(value: float, lo: float, hi: float) -> float:
    if value is None or np.isnan(value) or hi == lo:
        return 0.0
    x = (value - lo) / (hi - lo)
    return float(max(0.0, min(1.0, x)))

def pct_change(series: pd.Series, n: int) -> float:
    s = to_series_1d(series).dropna()
    if len(s) <= n:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[-n-1] - 1.0) * 100.0)

def last_float(x) -> float:
    if isinstance(x, (pd.Series, pd.Index)):
        if len(x) == 0: return float("nan")
        try: return float(x.iloc[-1])
        except Exception: return float("nan")
    if isinstance(x, pd.DataFrame):
        if x.shape[0] == 0: return float("nan")
        try: return float(x.iloc[-1, 0])
        except Exception: return float("nan")
    try:
        arr = np.asarray(x).reshape(-1)
        return float(arr[-1])
    except Exception:
        return float("nan")

# ========== Data fetch (cached) ==========
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_price_data(ticker: str):
    data = yf.download(ticker, period="10y", interval="1d", auto_adjust=True, progress=False, threads=False)
    if data is None or data.empty:
        raise ValueError(f"No price data for '{ticker}'.")
    close = to_series_1d(data["Close"], index=data.index).dropna()
    vol = to_series_1d(data["Volume"], index=data.index).dropna()
    close.index = pd.to_datetime(close.index)
    vol.index = pd.to_datetime(vol.index)
    return close, vol

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fundamentals(ticker: str):
    tk = yf.Ticker(ticker)
    q_earn = tk.quarterly_earnings
    q_fin = tk.quarterly_financials
    if not isinstance(q_earn, pd.DataFrame): q_earn = pd.DataFrame()
    if not isinstance(q_fin, pd.DataFrame): q_fin = pd.DataFrame()
    return q_earn, q_fin

def compute_features(ticker: str):
    close, vol = fetch_price_data(ticker)

    # --- Technicals
    last_price = last_float(close)
    sma200_series = close.rolling(200).mean() if len(close) >= 5 else pd.Series([last_price], index=close.index)
    sma200 = last_float(sma200_series)

    rsi14 = last_float(rsi(close, 14))
    macd_line, signal_line, _ = macd(close)
    macd_bull = bool(last_float(macd_line) > last_float(signal_line))

    vol30 = vol.rolling(30).mean()
    vol_spike = float(last_float(vol) / last_float(vol30)) if not np.isnan(last_float(vol30)) and last_float(vol30) > 0 else 1.0

    mom_1m = pct_change(close, 21)
    mom_3m = pct_change(close, 63)

    # --- Fundamentals (best-effort via yfinance)
    q_earn, q_fin = fetch_fundamentals(ticker)

    earnings_mom = float("nan")
    rev_accel = float("nan")
    gross_margin_exp = float("nan")

    # YoY revenue proxy + acceleration
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
    except Exception:
        pass

    # Gross margin expansion (latest - prior)
    try:
        if not q_fin.empty and "Total Revenue" in q_fin.index and "Gross Profit" in q_fin.index:
            rev_row = pd.to_numeric(q_fin.loc["Total Revenue"], errors="coerce").dropna()
            gp_row = pd.to_numeric(q_fin.loc["Gross Profit"], errors="coerce").dropna()
            common_cols = rev_row.index.intersection(gp_row.index)
            gm = (gp_row[common_cols] / rev_row[common_cols] * 100.0).dropna()
            if len(gm) >= 2:
                gross_margin_exp = float(gm.iloc[0] - gm.iloc[1])
    except Exception:
        pass

    return dict(
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

# ========== Weights (Hybrid: Tech 60 / Fund 40) ==========
W_FUND = {"earnings_mom": 16, "rev_accel": 12, "gross_margin_exp": 12}  # total 40
W_TECH = {"sma200_breakout": 18, "rsi_zone": 12, "macd_bull": 10, "vol_spike": 10, "momentum_combo": 10}  # total 60

# ========== Scoring ==========
def score_fundamentals(f):
    rows = []

    em = f.get("earnings_mom", float("nan"))
    if np.isnan(em): s=0; note="Missing/insufficient revenue history"
    else:  # widen to −20…+50%
        s=int(round(zscore(em, -20, 50)*100)); note=f"YoY revenue Δ={em:.1f}% (−20…+50 → 0…100)"
    rows.append(("Earnings Momentum (YoY revenue)", em if not np.isnan(em) else "N/A", s, W_FUND["earnings_mom"], note))

    ra = f.get("rev_accel", float("nan"))
    if np.isnan(ra): s=0; note="Need ≥3 quarters"
    else:  # widen to −15…+25 pp
        s=int(round(zscore(ra, -15, 25)*100)); note=f"Accel QoQ={ra:.1f} pp (−15…+25 → 0…100)"
    rows.append(("Revenue Acceleration", ra if not np.isnan(ra) else "N/A", s, W_FUND["rev_accel"], note))

    gm = f.get("gross_margin_exp", float("nan"))
    if np.isnan(gm): s=0; note="Gross margin history unavailable"
    else:  # widen to −8…+15 pp
        s=int(round(zscore(gm, -8, 15)*100)); note=f"GM Δ={gm:.1f} pp (−8…+15 → 0…100)"
    rows.append(("Gross Margin Expansion", gm if not np.isnan(gm) else "N/A", s, W_FUND["gross_margin_exp"], note))

    return rows

def score_technicals(f):
    rows = []

    # 200‑DMA breakout: −8…+15%
    pct = float("nan")
    if f.get("sma200", 0) and not np.isnan(f["sma200"]):
        pct = (f["price"]/f["sma200"] - 1.0) * 100.0
    s_dma = int(round(zscore(pct, -8, 15) * 100)) if not np.isnan(pct) else 0
    rows.append(("200‑Day Breakout", f"{pct:.1f}% vs 200‑DMA" if not np.isnan(pct) else "N/A",
                 s_dma, W_TECH["sma200_breakout"], "−8…+15% window rewards emerging breakouts"))

    # RSI: 45…72 (best ≈ 55–68)
    r = f.get("rsi14", float("nan"))
    s_rsi = int(round(zscore(r, 45, 72) * 100)) if not np.isnan(r) else 0
    rows.append(("RSI (14)", f"{r:.1f}" if not np.isnan(r) else "N/A",
                 s_rsi, W_TECH["rsi_zone"], "Best when rising through ~55–68"))

    # MACD bull
    mb = bool(f.get("macd_bull", False))
    s_macd = 100 if mb else 0
    rows.append(("MACD Bullish Cross", "Yes" if mb else "No", s_macd, W_TECH["macd_bull"], "MACD line > signal"))

    # Volume spike: 1.1×…1.8×
    vs = f.get("vol_spike", float("nan"))
    s_vol = int(round(zscore(vs, 1.1, 1.8) * 100)) if not np.isnan(vs) else 0
    rows.append(("Volume vs 30‑day avg", f"{vs:.2f}×" if not np.isnan(vs) else "N/A",
                 s_vol, W_TECH["vol_spike"], "Reward early expansion (≥1.1×)"))

    # Momentum graded: combine 1m & 3m (−15…+25%)
    m1 = f.get("mom_1m", float("nan"))
    m3 = f.get("mom_3m", float("nan"))
    s_m1 = int(round(zscore(m1, -15, 25) * 100)) if not np.isnan(m1) else 0
    s_m3 = int(round(zscore(m3, -15, 25) * 100)) if not np.isnan(m3) else 0
    s_mom = int(round((s_m1 + s_m3) / 2))
    val_mom = f"1m={m1:.1f}%, 3m={m3:.1f}%" if (not np.isnan(m1) and not np.isnan(m3)) else "N/A"
    rows.append(("Momentum (1m & 3m)", val_mom, s_mom, W_TECH["momentum_combo"],
                 "Graded: −15…+25% window; both months averaged"))
    return rows

def weighted_score(rows):
    total_w = sum(w for _,_,_,w,_ in rows)
    if total_w <= 0: return 0
    return int(round(sum(s * w for _,_,s,w,_ in rows) / total_w))

# ========== UI ==========
ticker = st.text_input("Ticker", value="ASTS").strip().upper()
if st.button("Run Screener"):
    try:
        with st.spinner("Fetching data & computing signals…"):
            feats = compute_features(ticker)
            fund_rows = score_fundamentals(feats)
            tech_rows = score_technicals(feats)

            # Use the actual weights across both sections
            final_score = weighted_score(fund_rows + tech_rows)

        st.subheader(f"Result for {ticker}")
        st.metric("Breakout Score (0–100)", final_score)
        if final_score >= 70:
            st.success("🔥 Strong Breakout Setup")
        elif final_score >= 50:
            st.info("🟡 Constructive / Watchlist")
        else:
            st.warning("🔴 Weak / Avoid")

        st.markdown("### Fundamentals Breakdown")
        st.dataframe(pd.DataFrame([{
            "Factor": n, "Value": v, "Subscore (0‑100)": s, "Weight": w, "Notes": note
        } for (n,v,s,w,note) in fund_rows]), use_container_width=True)

        st.markdown("### Technicals Breakdown")
        st.dataframe(pd.DataFrame([{
            "Factor": n, "Value": v, "Subscore (0‑100)": s, "Weight": w, "Notes": note
        } for (n,v,s,w,note) in tech_rows]), use_container_width=True)

        with st.expander("Debug / raw features"):
            st.json(feats)

        st.caption("Financial fields are best‑effort from yfinance. We can add premium feeds (DCF, insiders, short interest, options) next.")

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        st.caption("If this persists, try another ticker or retry in a minute.")