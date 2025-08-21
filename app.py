import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Breakout AI", layout="centered")
st.title("🚀 Breakout AI")
st.caption("Three modes: Pre‑Breakout (predictive), Breakout (trader), Growth (fundamental). Any ticker. Live data via yfinance.")

# ----------------- helpers -----------------
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

def zscore_to_score(current: float, hist: pd.Series, cap_sigma: float = 2.0):
    """Return (score 0..100, z). score = 50 + 25*z (clipped). Neutral=55 if missing."""
    h = to_series_1d(hist).dropna()
    if not np.isfinite(current) or h.empty: 
        return 55, float("nan")
    mu, sd = float(h.mean()), float(h.std(ddof=0))
    if sd == 0: return 55, 0.0
    z = (current - mu) / sd
    z = max(-cap_sigma, min(cap_sigma, z))
    score = int(round(max(0, min(100, 50 + 25*z))))
    return score, z

def softmax(scores, tau=8.0):
    s = np.array(scores, dtype=float)
    exps = np.exp((s - 70.0)/max(tau,1e-6))  # anchor around 70
    w = exps/np.sum(exps) if np.sum(exps)>0 else np.ones_like(s)/len(s)
    return w

def softmax_aggregate(rows, tau=8.0):
    scores = np.array([s for (_,_,s,_,_) in rows], dtype=float)
    if len(scores)==0: return 0
    w = softmax(scores, tau=tau)
    return int(round(np.sum(scores*w)))

def minmax_score(v, lo, hi, neutral=55):
    if not np.isfinite(v): return neutral
    x = (v - lo) / (hi - lo)
    return int(round(100 * max(0, min(1, x))))

# ----------------- data -----------------
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
    except Exception: 
        bull = False
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

    # fundamentals (for Growth mode)
    tkr = yf.Ticker(ticker)
    q_earn = tkr.quarterly_earnings if isinstance(tkr.quarterly_earnings,pd.DataFrame) else pd.DataFrame()
    q_fin  = tkr.quarterly_financials if isinstance(tkr.quarterly_financials,pd.DataFrame) else pd.DataFrame()
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

    # relative strength vs SPY (3m) + history
    spy_close, _ = fetch_prices("SPY")
    rs_3m = float("nan"); rs_3m_hist = pd.Series(dtype=float)
    try:
        spy_aligned = spy_close.reindex(close.index).ffill()
        rs_series = (close.pct_change(63) - spy_aligned.pct_change(63))*100.0
        rs_3m = last_float(rs_series); rs_3m_hist = rs_series.tail(N)
    except Exception: pass

    # also return raw series for predictive mode
    return dict(
        price=price, sma200=sma200, pct_to_200=pct_to_200,
        rsi14=rsi14, macd_bull=macd_bull, vol_spike=vol_spike,
        mom_1m=mom_1m, mom_3m=mom_3m, rs_3m=rs_3m,
        pct_to_200_hist=pct_to_200_hist, rsi_hist=rsi_hist, volspike_hist=volspike_hist,
        mom1_hist=mom1_hist, mom3_hist=mom3_hist, rs_3m_hist=rs_3m_hist,
        earnings_mom=earnings_mom, rev_accel=rev_accel, gross_margin_exp=gm_exp,
        _close=close, _vol=vol, _spy=spy_close
    )

# ----------------- scoring: BREAKOUT & GROWTH -----------------
def breakout_scoring(f):
    """Pure breakout: technicals only, confluence + downtrend penalties."""
    TECH_FLOOR = 45

    s_pct200, z_pct200 = zscore_to_score(f["pct_to_200"], f["pct_to_200_hist"]); s_pct200=max(TECH_FLOOR,s_pct200)
    s_rsi,   z_rsi    = zscore_to_score(f["rsi14"],      f["rsi_hist"]);         s_rsi=max(TECH_FLOOR,s_rsi)
    s_vol,   z_vol    = zscore_to_score(f["vol_spike"],  f["volspike_hist"]);    s_vol=max(TECH_FLOOR,s_vol)
    s_m1,_            = zscore_to_score(f["mom_1m"],     f["mom1_hist"])
    s_m3,_            = zscore_to_score(f["mom_3m"],     f["mom3_hist"])
    s_mom = int(round((max(TECH_FLOOR,s_m1)+max(TECH_FLOOR,s_m3))/2))
    s_rs,   z_rs     = zscore_to_score(f["rs_3m"],       f["rs_3m_hist"]);       s_rs=max(TECH_FLOOR,s_rs)

    tech_rows = [
        ("Pct vs 200‑DMA (z)", f"{f['pct_to_200']:.1f}%", s_pct200, 30, f"z={z_pct200:.2f}σ"),
        ("Volume/30d (z)",     f"{f['vol_spike']:.2f}×",   s_vol,    25, f"z={z_vol:.2f}σ"),
        ("RSI(14) (z)",        f"{f['rsi14']:.1f}",       s_rsi,    20, f"z={z_rsi:.2f}σ"),
        ("Rel.Str. vs SPY (3m z)", f"{f['rs_3m']:.1f}%",  s_rs,     15, f"z={z_rs:.2f}σ"),
        ("Momentum (1m & 3m z)",   f"1m={f['mom_1m']:.1f}%, 3m={f['mom_3m']:.1f}%", s_mom, 10, "avg of z‑scores"),
    ]

    conf_hits = sum(1 for (_,_,s,_,_) in tech_rows if s>=70)
    conf_bonus = 15 if conf_hits >= 2 else 0

    penalties = 0
    if np.isfinite(f["pct_to_200"]) and f["pct_to_200"] < 0: penalties -= 20
    if np.isfinite(f["rs_3m"]) and f["rs_3m"] < 0: penalties -= 10
    if np.isfinite(f["mom_1m"]) and f["mom_1m"] < 0: penalties -= 10

    tech_soft = softmax_aggregate(tech_rows, tau=6.0)
    base = int(round(tech_soft + penalties))
    return [], tech_rows, 0, tech_soft, conf_bonus, conf_hits, penalties

def growth_scoring(f):
    """Growth/fundamental mode: fundamentals heavy, technicals capped."""
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
    tech_soft = min(70, softmax_aggregate(tech_rows, tau=10.0))

    s_em = minmax_score(f.get("earnings_mom",np.nan), -20, 50)
    s_ra = minmax_score(f.get("rev_accel",np.nan),     -15, 25)
    s_gm = minmax_score(f.get("gross_margin_exp",np.nan), -8, 15)
    fund_rows = [
        ("Earnings Momentum (YoY revenue)", f.get("earnings_mom","N/A"), s_em, 10, "−20…+50% mapped"),
        ("Revenue Acceleration (pp)",       f.get("rev_accel","N/A"),     s_ra,  5, "−15…+25 pp"),
        ("Gross Margin Expansion (pp)",     f.get("gross_margin_exp","N/A"), s_gm, 5, "−8…+15 pp"),
    ]
    fw = sum(w for *_ , w, _ in fund_rows)
    fund_avg = int(round(sum(s*w for (_,_,s,w,_) in fund_rows)/fw)) if fw>0 else 0

    base = int(round(0.70 * fund_avg + 0.30 * tech_soft))
    return fund_rows, tech_rows, fund_avg, tech_soft, 0, 0, 0

# ----------------- NEW: Pre‑Breakout (predictive) -----------------
def prebreakout_scoring(ticker: str, f: dict):
    """
    Predictive: score setup quality BEFORE the pop.
    Signals:
      - Tight base / volatility contraction (20d volatility percentile low)
      - Bollinger band width (20d) percentile low
      - Volume dry‑up (10d/50d)
      - Proximity to pivot (≤3% below recent base high)
      - RS line leading SPY (RS near 1y high while price still below pivot)
      - OBV slope positive (20d)
    """
    close = f["_close"]; vol = f["_vol"]; spy = f["_spy"]
    N = min(len(close), 252)
    ch = close.tail(N); vh = vol.tail(N)

    # 1) Volatility contraction (20d std of returns)
    ret = ch.pct_change()
    vol20 = ret.rolling(20).std()*100.0
    vol20_pctile = int(round(100*(vol20.dropna() <= last_float(vol20)).mean())) if len(vol20.dropna())>0 else 55
    s_vol_contract = 100 - vol20_pctile  # lower vol = tighter = better

    # 2) Bollinger Band Width (20d)
    ma20 = ch.rolling(20).mean()
    std20 = ch.rolling(20).std()
    bbw = (std20 / ma20).replace([np.inf,-np.inf], np.nan)
    bbw_pctile = int(round(100*(bbw.dropna() <= last_float(bbw)).mean())) if len(bbw.dropna())>0 else 55
    s_bbw = 100 - bbw_pctile  # tighter bands better

    # 3) Volume dry‑up (10d vs 50d)
    v10 = vh.rolling(10).mean(); v50 = vh.rolling(50).mean()
    vol_dryup_ratio = last_float(v10)/last_float(v50) if last_float(v50)>0 else 1.0
    s_dryup = minmax_score(1.0 - vol_dryup_ratio, -0.2, 0.4)  # reward <1.0; strong if <=0.6

    # 4) Define base (last ~8 weeks) and pivot
    base_win = 40  # ~8 weeks
    base = ch.tail(base_win)
    if len(base)<10:
        pivot = last_float(ch.rolling(10).max())
    else:
        pivot = float(base.max())
    last_p = last_float(ch)
    dist_to_pivot = (pivot - last_p)/pivot*100.0 if pivot>0 else float("nan")
    # Good if within 0.5%..3% below pivot (not extended; ready)
    s_pivot_prox = minmax_score(3.0 - dist_to_pivot, -5.0, 3.0) if np.isfinite(dist_to_pivot) else 55

    # 5) RS line (stock/SPY) – is RS near 1y high while price still < pivot?
    spy_al = spy.reindex(close.index).ffill()
    rs_line = ch / spy_al.reindex(ch.index).ffill()
    rs_high_1y = rs_line.rolling(min(252,len(rs_line))).max()
    rs_at_high = last_float(rs_line) / last_float(rs_high_1y) if last_float(rs_high_1y)>0 else np.nan
    s_rs_lead = minmax_score(rs_at_high, 0.95, 1.02)  # near/new RS highs get rewarded

    # 6) OBV slope (20d)
    signed = np.sign(ch.diff().fillna(0.0))
    obv = (signed * vh).cumsum()
    obv_slope = (obv.diff(20).iloc[-1]) if len(obv)>=21 else np.nan
    # scale slope by recent avg volume to be dimensionless
    norm = last_float(v50)
    obv_norm = float(obv_slope / norm) if (np.isfinite(obv_slope) and norm>0) else np.nan
    s_obv = minmax_score(obv_norm, -0.5, 0.5)

    # Composite predictive score (weights sum to 100)
    pred_rows = [
        ("Volatility Contraction (20d pctile low)", f"{vol20.iloc[-1]:.2f}% std" if np.isfinite(last_float(vol20)) else "N/A", s_vol_contract, 25, "Tighter = better"),
        ("Band Width (20d pctile low)", f"{bbw.iloc[-1]:.4f}" if np.isfinite(last_float(bbw)) else "N/A", s_bbw, 20, "Tighter bands = coiled"),
        ("Volume Dry‑Up (10d/50d)", f"{vol_dryup_ratio:.2f}×", s_dryup, 20, "Lower than 1.0 preferred"),
        ("Proximity to Pivot", f"{dist_to_pivot:.2f}% below", s_pivot_prox, 20, "0.5–3% below pivot is ideal"),
        ("RS Line Leading (vs SPY)", f"{rs_at_high:.3f}× of 1y high" if np.isfinite(rs_at_high) else "N/A", s_rs_lead, 10, "Near RS highs pre‑breakout"),
        ("OBV Slope (20d)", f"{obv_norm:.3f}" if np.isfinite(obv_norm) else "N/A", s_obv, 5, "Accumulation"),
    ]
    pred_score = int(round(sum(s*w for (_,_,s,w,_) in pred_rows)/sum(w for *_ , w, _ in pred_rows)))

    # Status tag
    if np.isfinite(dist_to_pivot) and dist_to_pivot <= 0:
        status = "Early Breakout (above pivot)"
    elif np.isfinite(dist_to_pivot) and 0 < dist_to_pivot <= 3.0:
        status = "Setup (near pivot)"
    elif np.isfinite(dist_to_pivot) and dist_to_pivot <= 6.0:
        status = "Watchlist (within 6%)"
    else:
        status = "Forming Base / Too Far"

    extra = {
        "pivot_price": float(pivot) if np.isfinite(pivot) else None,
        "distance_to_pivot_%": float(dist_to_pivot) if np.isfinite(dist_to_pivot) else None,
        "rs_near_1y_high_ratio": float(rs_at_high) if np.isfinite(rs_at_high) else None
    }
    return pred_rows, pred_score, status, extra

# ----------------- UI -----------------
mode = st.radio("Mode", ["Pre‑Breakout (predictive)","Breakout (trader)","Growth (fundamental)"], horizontal=True)
ticker = st.text_input("Ticker", value="TSLA").strip().upper()

if st.button("Run Screener"):
    try:
        with st.spinner("Fetching & scoring…"):
            feats = compute_features(ticker)
            mr_bonus, regime_flags = market_regime_bonus()

            if mode.startswith("Pre‑Breakout"):
                pred_rows, pred_score, status, extra = prebreakout_scoring(ticker, feats)
                final_score = int(min(100, max(0, pred_score + mr_bonus)))
                st.subheader(f"{mode} — {ticker}")
                st.metric("Pre‑Breakout Setup Score (0–100)", final_score)
                if final_score >= 70: st.success(f"✅ {status}")
                elif final_score >= 50: st.info(f"🟡 {status}")
                else: st.warning(f"🔴 {status}")
                st.markdown("### Predictive Breakdown")
                st.dataframe(pd.DataFrame([{"Factor":n,"Value":v,"Subscore (0‑100)":s,"Weight":w,"Notes":note}
                                           for (n,v,s,w,note) in pred_rows]), use_container_width=True)
                st.markdown("### Market / Extras")
                st.json({"Market bonus": f"+{mr_bonus}", **regime_flags, **extra})

            elif mode.startswith("Breakout"):
                fund_rows, tech_rows, fund_avg, tech_soft, conf_bonus, conf_hits, penalties = breakout_scoring(feats)
                final_score = int(min(100, max(0, tech_soft + conf_bonus + penalties + mr_bonus)))
                st.subheader(f"{mode} — {ticker}")
                st.metric("Breakout Score (0–100)", final_score)
                if final_score >= 70: st.success("🔥 Strong Breakout Setup")
                elif final_score >= 50: st.info("🟡 Constructive / Watchlist")
                else: st.warning("🔴 Weak / Avoid")
                st.markdown("### Technicals Breakdown")
                st.dataframe(pd.DataFrame([{"Factor":n,"Value":v,"Subscore (0‑100)":s,"Weight":w,"Notes":note}
                                           for (n,v,s,w,note) in tech_rows]), use_container_width=True)
                st.markdown("### Bonuses / Penalties")
                st.json({
                    "Confluence bonus": f"+{conf_bonus} (tech factors ≥70: {conf_hits})",
                    "Downtrend penalties": penalties,
                    "Market regime bonus": f"+{mr_bonus}",
                    **regime_flags
                })

            else:  # Growth
                fund_rows, tech_rows, fund_avg, tech_soft, _, _, _ = growth_scoring(feats)
                final_score = int(min(100, max(0, int(round(0.70*fund_avg + 0.30*tech_soft)) + mr_bonus)))
                st.subheader(f"{mode} — {ticker}")
                st.metric("Growth Score (0–100)", final_score)
                if final_score >= 70: st.success("✅ Strong Growth Profile")
                elif final_score >= 50: st.info("🟡 Mixed / Watch")
                else: st.warning("🔴 Weak / Avoid")
                st.markdown("### Technicals Breakdown")
                st.dataframe(pd.DataFrame([{"Factor":n,"Value":v,"Subscore (0‑100)":s,"Weight":w,"Notes":note}
                                           for (n,v,s,w,note) in tech_rows]), use_container_width=True)
                st.markdown("### Fundamentals Breakdown")
                st.dataframe(pd.DataFrame([{"Factor":n,"Value":v,"Subscore (0‑100)":s,"Weight":w,"Notes":note}
                                           for (n,v,s,w,note) in fund_rows]), use_container_width=True)
                st.markdown("### Market")
                st.json({"Market regime bonus": f"+{mr_bonus}", **regime_flags})

        with st.expander("Debug / raw features"):
            st.json({k:(float(v) if isinstance(v,(int,float,np.floating)) else (None if v is None else str(v)))
                     for k,v in feats.items()
                     if k not in ["pct_to_200_hist","rsi_hist","volspike_hist","mom1_hist","mom3_hist","rs_3m_hist","_close","_vol","_spy"]})

        st.caption("Pre‑Breakout mode focuses on *setup quality*: tightness, dry‑up, pivot proximity, and RS leadership — to predict the move, not chase it.")

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        st.caption("Try another ticker or retry shortly.")