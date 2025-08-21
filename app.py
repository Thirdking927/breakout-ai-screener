import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Breakout AI", layout="centered")
st.title("🚀 Breakout AI")
st.caption("Switch between Breakout (trader) and Growth (fundamental) modes. Any ticker. Live data via yfinance.")

# ---------- helpers ----------
def to_series_1d(x, index=None) -> pd.Series:
    if isinstance(x, pd.Series): return x
    if isinstance(x, pd.DataFrame): return x.iloc[:,0] if x.shape[1] else pd.Series(dtype=float)
    arr = np.asarray(x).reshape(-1)
    return pd.Series(arr, index=index) if index is not None else pd.Series(arr)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = to_series_1d(series).dropna()
    d = s.diff()
    gain = (d.where(d>0,0)).rolling(period).mean()
    loss = (-d.where(d<0,0)).rolling(period).mean()
    rs = gain / loss.replace(0,np.nan)
    return 100 - (100/(1+rs))

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
    return float((s.iloc[-1]/s.iloc[-n-1]-1.0)*100.0)

def last_float(x) -> float:
    try:
        if isinstance(x,(pd.Series,pd.Index)): return float(x.iloc[-1]) if len(x) else float("nan")
        if isinstance(x,pd.DataFrame): return float(x.iloc[-1,0]) if len(x) else float("nan")
        arr = np.asarray(x).reshape(-1); return float(arr[-1])
    except Exception: return float("nan")

def zscore_to_score(current: float, hist: pd.Series, cap_sigma: float = 2.0) -> tuple[int,float]:
    h = to_series_1d(hist).dropna()
    if not np.isfinite(current) or h.empty: return 55, float("nan")
    mu, sd = float(h.mean()), float(h.std(ddof=0))
    if sd == 0: return 55, 0.0
    z = (current-mu)/sd
    z = max(-cap_sigma, min(cap_sigma, z))
    score = int(round(max(0, min(100, 50 + 25*z))))
    return score, z

def weighted(rows):
    tw = sum(w for *_ , w, _ in rows)
    if tw <= 0: return 0
    return int(round(sum(s*w for *_, s, w, _ in rows)/tw))

def softmax(scores, tau=10.0):
    s = np.array(scores, dtype=float)
    exps = np.exp((s-70.0)/max(tau,1e-6))
    w = exps/np.sum(exps) if np.sum(exps)>0 else np.ones_like(s)/len(s)
    return w

def softmax_aggregate(rows, tau=10.0):
    scores = np.array([s for _,_,s,_,_ in rows], dtype=float)
    if len(scores)==0: return 0
    w = softmax(scores, tau=tau)
    return int(round(np.sum(scores*w)))

# ---------- data ----------
@st.cache_data(ttl=1800)
def fetch_prices(ticker: str):
    df = yf.download(ticker, period="10y", interval="1d", auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty: raise ValueError(f"No price data for '{ticker}'.")
    close = to_series_1d(df["Close"], index=df.index).dropna()
    vol   = to_series_1d(df["Volume"], index=df.index).dropna()
    close.index = pd.to_datetime(close.index); vol.index = pd.to_datetime(vol.index)
    return close, vol

@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker: str):
    t = yf.Ticker(ticker)
    q_earn = t.quarterly_earnings if isinstance(t.quarterly_earnings,pd.DataFrame) else pd.DataFrame()
    q_fin  = t.quarterly_financials if isinstance(t.quarterly_financials,pd.DataFrame) else pd.DataFrame()
    return q_earn, q_fin

@st.cache_data(ttl=900)
def market_regime_bonus():
    try:
        spy, _ = fetch_prices("SPY")
        bull = last_float(spy) > last_float(spy.rolling(200).mean())
    except Exception: bull = False
    vix = yf.download("^VIX", period="1y", interval="1d", progress=False, threads=False)
    vix_last = float(vix["Close"].iloc[-1]) if isinstance(vix,pd.DataFrame) and not vix.empty else float("nan")
    calm = (vix_last < 20) if np.isfinite(vix_last) else False
    return (5 if bull else 0) + (3 if calm else 0), {"SPY>200DMA": bull, "VIX<20": calm}

def compute_features(ticker: str):
    close, vol = fetch_prices(ticker)
    N = min(len(close), 252)
    c_hist = close.tail(N); v_hist = vol.tail(N)

    price = last_float(close)
    sma200_series = close.rolling(200).mean() if len(close)>=200 else close.rolling(min(len(close),50)).mean()
    sma200 = last_float(sma200_series)
    pct_to_200 = (price/sma200 - 1.0)*100.0 if np.isfinite(price) and np.isfinite(sma200) and sma200>0 else float("nan")

    rsi_series = rsi(close,14); rsi14 = last_float(rsi_series)
    macd_line, sig_line, _ = macd(close); macd_bull = bool(last_float(macd_line) > last_float(sig_line))

    vol30 = vol.rolling(30).mean()
    vol_spike = float(last_float(vol)/last_float(vol30)) if np.isfinite(last_float(vol30)) and last_float(vol30)>0 else 1.0

    mom_1m = pct_change(close,21); mom_3m = pct_change(close,63)

    # history for scaling
    pct_to_200_hist = ((c_hist/(c_hist.rolling(200).mean()))-1.0)*100.0 if len(c_hist)>=200 else ((c_hist/(c_hist.rolling(50).mean()))-1.0)*100.0
    rsi_hist = rsi_series.tail(N)
    volspike_hist = (v_hist/v_hist.rolling(30).mean())
    mom1_hist = c_hist.pct_change(21)*100.0
    mom3_hist = c_hist.pct_change(63)*100.0

    # fundamentals (used only in Growth mode)
    q_earn, q_fin = fetch_fundamentals(ticker)
    earnings_mom = rev_accel = gm_exp = float("nan")
    try:
        if not q_earn.empty and "Revenue" in q_earn.columns:
            q = q_earn.tail(6).copy(); rev = pd.to_numeric(q["Revenue"], errors="coerce").dropna()
            if len(rev)>=5: earnings_mom = float((rev.iloc[-1]/rev.iloc[-5]-1.0)*100.0)
            elif len(rev)>=2: earnings_mom = float((rev.iloc[-1]/rev.iloc[-2]-1.0)*100.0)
            if len(rev)>=3:
                g1 = (rev.iloc[-1]/rev.iloc[-2]-1.0)*100.0
                g2 = (rev.iloc[-2]/rev.iloc[-3]-1.0)*100.0
                rev_accel = float(g1-g2)
    except Exception: pass
    try:
        if not q_fin.empty and "Total Revenue" in q_fin.index and "Gross Profit" in q_fin.index:
            rev_row = pd.to_numeric(q_fin.loc["Total Revenue"], errors="coerce").dropna()
            gp_row  = pd.to_numeric(q_fin.loc["Gross Profit"], errors="coerce").dropna()
            common  = rev_row.index.intersection(gp_row.index)
            gm = (gp_row[common]/rev_row[common]*100.0).dropna()
            if len(gm)>=2: gm_exp = float(gm.iloc[0]-gm.iloc[1])
    except Exception: pass

    # relative strength vs SPY (3m) + its history
    spy_close, _ = fetch_prices("SPY")
    rs_3m = float("nan"); rs_3m_hist = pd.Series(dtype=float)
    try:
        spy_aligned = spy_close.reindex(close.index).ffill()
        rs_series = (close.pct_change(63) - spy_aligned.pct_change(63))*100.0
        rs_3m = last_float(rs_series); rs_3m_hist = rs_series.tail(N)
    except Exception: pass

    return dict(
        price=price, sma200=sma200, pct_to_200=pct_to_200,
        rsi14=rsi14, macd_bull=macd_bull, vol_spike=vol_spike,
        mom_1m=mom_1m, mom_3m=mom_3m, rs_3m=rs_3m,
        pct_to_200_hist=pct_to_200_hist, rsi_hist=rsi_hist, volspike_hist=volspike_hist,
        mom1_hist=mom1_hist, mom3_hist=mom3_hist, rs_3m_hist=rs_3m_hist,
        earnings_mom=earnings_mom, rev_accel=rev_accel, gross_margin_exp=gm_exp
    )

# ---------- scorers ----------
def breakout_mode_scores(f):
    """Pure breakout: heavy technicals similar to your early version."""
    TECH_FLOOR = 45
    # z‑score based subscores
    s_pct200, z1 = zscore_to_score(f["pct_to_200"], f["pct_to_200_hist"]); s_pct200=max(TECH_FLOOR,s_pct200)
    s_rsi,   z2 = zscore_to_score(f["rsi14"],    f["rsi_hist"]);           s_rsi=max(TECH_FLOOR,s_rsi)
    s_vol,   z3 = zscore_to_score(f["vol_spike"],f["volspike_hist"]);      s_vol=max(TECH_FLOOR,s_vol)
    s_m1, _  = zscore_to_score(f["mom_1m"],      f["mom1_hist"])
    s_m3, _  = zscore_to_score(f["mom_3m"],      f["mom3_hist"])
    s_mom = int(round((max(TECH_FLOOR,s_m1)+max(TECH_FLOOR,s_m3))/2))
    s_rs,   z4 = zscore_to_score(f["rs_3m"],     f["rs_3m_hist"]);         s_rs=max(TECH_FLOOR,s_rs)

    tech_rows = [
        ("Pct vs 200‑DMA (z)", f"{f['pct_to_200']:.1f}%", s_pct200, 30, f"z={z1:.2f}σ"),
        ("Volume/30d (z)",     f"{f['vol_spike']:.2f}×",   s_vol,    25, f"z={z3:.2f}σ"),
        ("RSI(14) (z)",        f"{f['rsi14']:.1f}",       s_rsi,    20, f"z={z2:.2f}σ"),
        ("Rel.Str. vs SPY (3m z)", f"{f['rs_3m']:.1f}%",  s_rs,     15, f"z={z4:.2f}σ"),
        ("Momentum (1m & 3m z)",   f"1m={f['mom_1m']:.1f}%, 3m={f['mom_3m']:.1f}%", s_mom, 10, "avg of z‑scores"),
    ]
    # confluence
    conf_hits = sum(1 for *_ , s, _, _ in [(r[0],r[1],r[2],r[3],r[4]) for r in tech_rows] if s>=70)
    conf_bonus = 10 if conf_hits>=3 else 0
    # aggregate technicals with softmax (emphasize winners)
    tech_soft = softmax_aggregate(tech_rows, tau=8.0)
    # no fundamentals in breakout mode
    fund_rows, fund_avg = [], 0
    return fund_rows, tech_rows, fund_avg, tech_soft, conf_bonus, conf_hits

def growth_mode_scores(f):
    """Growth/fundamental heavier mode (still technical‑led)."""
    # technicals (lighter weights than breakout)
    s_pct200,_ = zscore_to_score(f["pct_to_200"], f["pct_to_200_hist"])
    s_rsi,_    = zscore_to_score(f["rsi14"],      f["rsi_hist"])
    s_vol,_    = zscore_to_score(f["vol_spike"],  f["volspike_hist"])
    s_m1,_     = zscore_to_score(f["mom_1m"],     f["mom1_hist"])
    s_m3,_     = zscore_to_score(f["mom_3m"],     f["mom3_hist"])
    s_mom = int(round((s_m1+s_m3)/2))
    s_rs,_     = zscore_to_score(f["rs_3m"],      f["rs_3m_hist"])

    tech_rows = [
        ("Pct vs 200‑DMA (z)", f"{f['pct_to_200']:.1f}%", s_pct200, 20, ""),
        ("RSI(14) (z)",        f"{f['rsi14']:.1f}",       s_rsi,    15, ""),
        ("Volume/30d (z)",     f"{f['vol_spike']:.2f}×",  s_vol,    15, ""),
        ("Rel.Str. vs SPY (3m z)", f"{f['rs_3m']:.1f}%",  s_rs,     10, ""),
        ("Momentum (1m & 3m z)",   f"1m={f['mom_1m']:.1f}%, 3m={f['mom_3m']:.1f}%", s_mom, 10, ""),
    ]
    tech_soft = softmax_aggregate(tech_rows, tau=10.0)

    # fundamentals (light)
    def mm(v, lo, hi):
        if not np.isfinite(v): return 55
        x = (v-lo)/(hi-lo)
        return int(round(100*max(0,min(1,x))))
    s_em = mm(f.get("earnings_mom",np.nan), -20, 50)
    s_ra = mm(f.get("rev_accel",np.nan),     -15, 25)
    s_gm = mm(f.get("gross_margin_exp",np.nan), -8, 15)
    fund_rows = [
        ("Earnings Momentum (YoY revenue)", f.get("earnings_mom","N/A"), s_em, 10, "−20…+50% mapped"),
        ("Revenue Acceleration (pp)",       f.get("rev_accel","N/A"),     s_ra,  5, "−15…+25 pp"),
        ("Gross Margin Expansion (pp)",     f.get("gross_margin_exp","N/A"), s_gm, 5, "−8…+15 pp"),
    ]
    fund_avg = weighted(fund_rows)
    return fund_rows, tech_rows, fund_avg, tech_soft, 0, 0

# ---------- UI ----------
mode = st.radio("Mode", ["Breakout (trader)","Growth (fundamental)"], horizontal=True)
ticker = st.text_input("Ticker", value="ASTS").strip().upper()

if st.button("Run Screener"):
    try:
        with st.spinner("Fetching & scoring…"):
            feats = compute_features(ticker)
            if mode.startswith("Breakout"):
                fund_rows, tech_rows, fund_avg, tech_soft, conf_bonus, conf_hits = breakout_mode_scores(feats)
                base = int(round(0.95*tech_soft + 0.05*fund_avg))  # almost pure technical
            else:
                fund_rows, tech_rows, fund_avg, tech_soft, conf_bonus, conf_hits = growth_mode_scores(feats)
                base = int(round(0.80*tech_soft + 0.20*fund_avg))

            mr_bonus, regime_flags = market_regime_bonus()
            final_score = int(min(100, max(0, base + conf_bonus + mr_bonus)))

        st.subheader(f"{mode} — {ticker}")
        st.metric("Breakout Score (0–100)", final_score)
        if final_score >= 70:
            st.success("🔥 Strong Breakout Setup")
        elif final_score >= 50:
            st.info("🟡 Constructive / Watchlist")
        else:
            st.warning("🔴 Weak / Avoid")

        st.markdown("### Technicals Breakdown")
        st.dataframe(pd.DataFrame([{"Factor":n,"Value":v,"Subscore (0‑100)":s,"Weight":w,"Notes":note}
                                   for (n,v,s,w,note) in tech_rows]), use_container_width=True)

        st.markdown("### Fundamentals Breakdown")
        if fund_rows:
            st.dataframe(pd.DataFrame([{"Factor":n,"Value":v,"Subscore (0‑100)":s,"Weight":w,"Notes":note}
                                       for (n,v,s,w,note) in fund_rows]), use_container_width=True)
        else:
            st.write("_(Not used in Breakout mode)_")

        st.markdown("### Bonuses")
        st.json({
            "Confluence bonus": f"+{conf_bonus} (tech factors ≥70: {conf_hits})",
            "Market regime bonus": f"+{mr_bonus}",
            **regime_flags
        })

        with st.expander("Debug / raw features"):
            st.json({k:(float(v) if isinstance(v,(int,float,np.floating)) else (None if v is None else str(v)))
                     for k,v in feats.items()
                     if k not in ["pct_to_200_hist","rsi_hist","volspike_hist","mom1_hist","mom3_hist","rs_3m_hist"]})

        st.caption("Breakout mode: 200‑DMA, volume, RSI, relative strength, momentum (softmax + confluence). Growth mode adds lighter fundamentals.")

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        st.caption("Try another ticker or retry shortly.")