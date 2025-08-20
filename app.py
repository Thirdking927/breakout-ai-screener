import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Breakout AI — Phase 2", layout="centered")
st.title("🚀 Breakout AI — Phase 2")
st.caption("Type any ticker. Live data via yfinance. Shows a 0–100 Breakout Score with full factor breakdown.")

# ----------------------------
# Helpers: indicators & scoring
# ----------------------------
def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def zscore(value, lo, hi):
    if hi == lo: return 0.0
    x = (value - lo) / (hi - lo)
    return float(max(0.0, min(1.0, x)))

def pct_change(series: pd.Series, n: int):
    if len(series) <= n: return np.nan
    return float((series.iloc[-1] / series.iloc[-n-1] - 1.0) * 100.0)

# ----------------------------
# Data fetch (cached)
# ----------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_price_data(ticker: str):
    data = yf.download(ticker, period="10y", interval="1d", auto_adjust=True, progress=False, threads=False)
    if data is None or data.empty:
        raise ValueError(f"No price data for '{ticker}'.")
    close = data["Close"].dropna()
    vol = data["Volume"].dropna()
    close.index = pd.to_datetime(close.index)
    vol.index = pd.to_datetime(vol.index)
    return close, vol

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_fundamentals(ticker: str):
    tk = yf.Ticker(ticker)

    # Quarterly earnings (Revenue / Earnings)
    q_earn = tk.quarterly_earnings  # columns: Revenue, Earnings
    # Quarterly financials (rows like Total Revenue, Gross Profit, ...)
    q_fin = tk.quarterly_financials  # wide format, columns are quarters

    return q_earn, q_fin

def compute_features(ticker: str):
    close, vol = fetch_price_data(ticker)

    # Technicals
    last_price = float(close.iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())
    rsi14 = float(rsi(close, 14).iloc[-1])
    macd_line, signal_line, hist = macd(close)
    macd_bull = bool(macd_line.iloc[-1] > signal_line.iloc[-1])
    vol30 = vol.rolling(30).mean()
    vol_spike = float(vol.iloc[-1] / vol30.iloc[-1]) if vol30.iloc[-1] > 0 else 1.0
    mom_1m = pct_change(close, 21)
    mom_3m = pct_change(close, 63)

    # Fundamentals (best-effort from yfinance)
    q_earn, q_fin = fetch_fundamentals(ticker)

    # Earnings momentum (YoY revenue growth from quarterly_earnings if available)
    earnings_mom = np.nan
    rev_accel = np.nan
    gross_margin_exp = np.nan

    if isinstance(q_earn, pd.DataFrame) and not q_earn.empty:
        q = q_earn.tail(5)
        if "Revenue" in q.columns:
            rev = q["Revenue"].astype(float)
            # YoY growth for latest quarter vs quarter 4 periods ago (approx)
            if len(rev) >= 5:
                earnings_mom = float((rev.iloc[-1] / rev.iloc[-5] - 1.0) * 100.0)
            elif len(rev) >= 2:
                earnings_mom = float((rev.iloc[-1] / rev.iloc[-2] - 1.0) * 100.0)

            # Acceleration: recent QoQ growth minus prior QoQ growth (rough proxy)
            if len(rev) >= 3:
                g1 = (rev.iloc[-1] / rev.iloc[-2] - 1.0) * 100.0
                g2 = (rev.iloc[-2] / rev.iloc[-3] - 1.0) * 100.0
                rev_accel = float(g1 - g2)

    if isinstance(q_fin, pd.DataFrame) and not q_fin.empty:
        # gross margin expansion (latest GM% minus prior GM%)
        try:
            rev_row = q_fin.loc["Total Revenue"].astype(float)
            gp_row = q_fin.loc["Gross Profit"].astype(float)
            if len(rev_row.dropna()) >= 2 and len(gp_row.dropna()) >= 2:
                gm = (gp_row / rev_row * 100.0).dropna()
                if len(gm) >= 2:
                    gross_margin_exp = float(gm.iloc[0] - gm.iloc[1])  # q_fin columns often reverse-chronological
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
# Scoring weights (Phase 2)
# ----------------------------
# Fundamentals (50)
W_FUND = {
    "earnings_mom": 20,      # YoY revenue momentum proxy
    "rev_accel": 15,         # acceleration
    "gross_margin_exp": 15,  # margin expansion
    # DCF/Insider placeholders could be added later with data sources
}
# Technicals (50)
W_TECH = {
    "sma200_breakout": 15,   # above 200-day
    "rsi_zone": 10,          # 50-70 ideal
    "macd_bull": 10,         # MACD line > signal
    "vol_spike": 10,         # >1.3x 30-day avg
    "momentum_combo": 5,     # 1m & 3m > 0
}

def score_fundamentals(f):
    rows = []

    # Earnings momentum (YoY revenue change %)
    em = f["earnings_mom"]
    if np.isnan(em): s = 0; note="Missing quarterly revenue; score=0"
    else:
        s = int(round(zscore(em, -10, 40) * 100))  # -10%..+40% mapped to 0..100
        note = f"YoY revenue Δ={em:.1f}% (−10…+40% → 0…100)"
    rows.append(("Earnings Momentum (YoY revenue)", em if not np.isnan(em) else "N/A", s, W_FUND["earnings_mom"], note))

    # Revenue acceleration (recent QoQ minus prior QoQ)
    ra = f["rev_accel"]
    if np.isnan(ra): s = 0; note="Insufficient quarters; score=0"
    else:
        s = int(round(zscore(ra, -10, 20) * 100))  # −10..+20% → 0..100
        note = f"Accel QoQ={ra:.1f} pp (−10…+20 → 0…100)"
    rows.append(("Revenue Acceleration", ra if not np.isnan(ra) else "N/A", s, W_FUND["rev_accel"], note))

    # Gross margin expansion (pp change)
    gm = f["gross_margin_exp"]
    if np.isnan(gm): s = 0; note="No gross margin series; score=0"
    else:
        s = int(round(zscore(gm, -5, 10) * 100))  # −5..+10 pp → 0..100
        note = f"GM Δ={gm:.1f} pp (−5…+10 → 0…100)"
    rows.append(("Gross Margin Expansion", gm if not np.isnan(gm) else "N/A", s, W_FUND["gross_margin_exp"], note))

    return rows

def score_technicals(f):
    rows = []

    # 200-day breakout
    above = f["price"] > f["sma200"]
    pct = (f["price"]/f["sma200"] - 1.0) * 100.0 if f["sma200"] else 0.0
    s = int(round(zscore(pct, -5, 10) * 100))  # −5%..+10% above → 0..100
    rows.append(("200‑Day Breakout", f"{pct:.1f}% vs 200‑DMA", s, W_TECH["sma200_breakout"], "Higher above (up to ~+10%) scores better"))

    # RSI zone (50–70 best)
    r = f["rsi14"]
    s_rsi = int(round(zscore(r, 40, 70) * 100))
    rows.append(("RSI Zone (14)", f"{r:.1f}", s_rsi, W_TECH["rsi_zone"], "50–70 preferred; <40 weak, >80 overbought"))

    # MACD bull
    mb = f["macd_bull"]
    s_macd = 100 if mb else 0
    rows.append(("MACD Bullish Cross", "Yes" if mb else "No", s_macd, W_TECH["macd_bull"], "MACD line > signal"))

    # Volume spike vs 30‑day avg
    vs = f["vol_spike"]
    s_vol = int(round(zscore(vs, 1.0, 2.0) * 100))  # 1.0…2.0x avg
    rows.append(("Volume Spike (x30d avg)", f"{vs:.2f}×", s_vol, W_TECH["vol_spike"], "Breakouts with >1.3× volume are stronger"))

    # Momentum combo (1m & 3m > 0)
    combo = (f["mom_1m"] if not np.isnan(f["mom_1m"]) else -1) > 0 and (f["mom_3m"] if not np.isnan(f["mom_3m"]) else -1) > 0
    s_mom = 100 if combo else 0
    rows.append(("Momentum (1m & 3m)", f"1m={f['mom_1m']:.1f}%, 3m={f['mom_3m']:.1f}%", s_mom, W_TECH["momentum_combo"], "Both > 0% preferred"))

    return rows

def weighted_score(rows):
    # rows: list of (name, value, subscore0..100, weight, note)
    total_w = sum(w for _,_,_,w,_ in rows)
    if total_w == 0: return 0
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

            # Combine for the final weighted score (50/50 split by section)
            fund_score = weighted_score(fund_rows)
            tech_score = weighted_score(tech_rows)
            final_score = int(round((fund_score * 0.50) + (tech_score * 0.50)))

        # --- Top summary
        st.subheader(f"Result for {ticker}")
        st.metric("Breakout Score (0–100)", final_score)
        if final_score >= 75:
            st.success("🔥 Strong Breakout Setup")
        elif final_score >= 55:
            st.info("🟡 Constructive / Watchlist")
        else:
            st.warning("🔴 Weak / Avoid")

        # --- Detailed breakdown
        st.markdown("### Fundamentals Breakdown")
        fund_df = pd.DataFrame([{
            "Factor": n, "Value": v, "Subscore (0‑100)": s, "Weight": w, "Notes": note
        } for (n,v,s,w,note) in fund_rows])
        st.dataframe(fund_df, use_container_width=True)

        st.markdown("### Technicals Breakdown")
        tech_df = pd.DataFrame([{
            "Factor": n, "Value": v, "Subscore (0‑100)": s, "Weight": w, "Notes": note
        } for (n,v,s,w,note) in tech_rows])
        st.dataframe(tech_df, use_container_width=True)

        with st.expander("Debug / raw features"):
            st.json(feats)

        st.caption("Notes: Earnings/financial fields are best‑effort from yfinance. We can add premium feeds (insiders, short interest, options) later and fold them into this table with weights.")

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        st.caption("If this persists, try another ticker or retry in a minute.")