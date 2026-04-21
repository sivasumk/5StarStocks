import streamlit as st
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="5-Star Stocks Scanner", layout="wide")

BASE_DIR = Path(__file__).parent

# ─── Nifty 500 stock list (static CSV) ─────────────────────────
@st.cache_data
def fetch_nifty500():
    df = pd.read_csv(BASE_DIR / "nifty500.csv")
    return df["Symbol"].str.strip().tolist()


# ─── F&O stock list (static CSV) ───────────────────────────────
@st.cache_data
def fetch_fno_stocks():
    df = pd.read_csv(BASE_DIR / "fno_stocks.csv")
    return set(df["Symbol"].str.strip().tolist())


# ─── Download OHLCV (single chunk, no cache) ────────────────────
def download_chunk(symbols, interval):
    tickers = [s + ".NS" for s in symbols]
    period = "6mo" if interval == "1d" else "2y"
    data = yf.download(tickers, period=period, interval=interval,
                       group_by="ticker", threads=False, progress=False)
    return data


# ─── Signal computation ─────────────────────────────────────────
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_signals(data, symbols, interval, fno_set=None):
    band_len = 34
    slope_len = 5
    slope_lb = 3
    slope_thresh = 0.0
    exit_slope_rev = 0.0
    vol_len = 20
    spread_len = 20
    vol_thresh_high = 1.5
    vol_thresh_low = 0.7
    sl_atr_mult = 2.5
    atr_len = 14
    max_hold = 30
    cooldown = 3

    longs = []
    shorts = []

    for sym in symbols:
        ticker = sym + ".NS"
        try:
            if ticker in data.columns.get_level_values(0):
                df = data[ticker].dropna(subset=["Close"]).copy()
            else:
                continue
        except Exception:
            continue

        if len(df) < band_len + slope_lb + 5:
            continue

        c = df["Close"].values
        h = df["High"].values
        o = df["Open"].values
        l = df["Low"].values
        v = df["Volume"].values
        n = len(c)

        # Precompute EMAs and indicators as arrays
        ema_c = pd.Series(c).ewm(span=band_len, adjust=False).mean().values
        ema_h = pd.Series(h).ewm(span=band_len, adjust=False).mean().values
        ema_l = pd.Series(l).ewm(span=band_len, adjust=False).mean().values
        ema_sl = pd.Series(c).ewm(span=slope_len, adjust=False).mean().values

        slope_arr = [float('nan')] * n
        for j in range(slope_lb, n):
            if ema_sl[j - slope_lb] != 0:
                slope_arr[j] = (ema_sl[j] - ema_sl[j - slope_lb]) / ema_sl[j - slope_lb] * 100

        # ATR
        tr_vals = [0.0] * n
        for j in range(1, n):
            tr_vals[j] = max(h[j] - l[j], abs(h[j] - c[j-1]), abs(l[j] - c[j-1]))
        atr_arr = pd.Series(tr_vals).rolling(atr_len).mean().values

        # Volume & spread rolling averages
        avg_vol_arr = pd.Series(v).rolling(vol_len).mean().values
        spread_arr = h - l
        avg_spread_arr = pd.Series(spread_arr).rolling(spread_len).mean().values

        # ── State machine (mirrors Pine Script exactly) ──
        pos = 0           # 0=flat, 1=long, -1=short
        entry_price = 0.0
        sl_price = 0.0
        entry_bar = -999
        last_exit_bar = -999
        entry_score = 0
        entry_slope = 0.0
        entry_rel_vol = 0.0

        # VPA helper for bar j
        def vpa_at(j):
            av = avg_vol_arr[j]
            rel_v = v[j] / av if av > 0 else 1.0
            sp = spread_arr[j]
            avs = avg_spread_arr[j]
            rel_sp = sp / avs if avs > 0 else 1.0
            cp = (c[j] - l[j]) / sp if sp > 0 else 0.5
            hv = rel_v >= vol_thresh_high
            lv = rel_v <= vol_thresh_low
            ws = rel_sp >= 1.3
            ns = rel_sp <= 0.7
            bull = c[j] > o[j]
            bear = c[j] < o[j]
            bs = abs(c[j] - o[j])
            uw = h[j] - max(c[j], o[j])
            lw = min(c[j], o[j]) - l[j]
            climax_u = hv and ws and cp > 0.7 and bull
            climax_d = hv and ws and cp < 0.3 and bear
            nd = lv and ns and bull and cp < 0.5
            nsp = lv and ns and bear and cp > 0.5
            ut = hv and cp < 0.3 and uw > bs
            spr = hv and cp > 0.7 and lw > bs
            return rel_v, cp, nd, nsp, climax_u, climax_d, ut, spr

        # Walk bar-by-bar starting after warmup
        start = max(band_len + slope_lb, vol_len, spread_len, atr_len + 1)
        for j in range(start, n):
            sl_val = slope_arr[j]
            if pd.isna(sl_val) or pd.isna(atr_arr[j]):
                continue

            cl, op_j, hi, lo = c[j], o[j], h[j], l[j]
            eh, el_j, ec = ema_h[j], ema_l[j], ema_c[j]
            above_h = cl > eh
            below_l = cl < el_j
            is_bull = cl > op_j
            is_bear = cl < op_j

            rel_v, cp, nd, nsp, climax_u, climax_d, ut, spr = vpa_at(j)

            # Previous bar VPA lookback
            if j >= 1:
                _, _, nd_prev, nsp_prev, _, _, _, _ = vpa_at(j - 1)
            else:
                nd_prev, nsp_prev = False, False

            vpa_block_long = (nd or nd_prev) or climax_u
            vpa_block_short = (nsp or nsp_prev) or climax_d

            long_entry = sl_val > slope_thresh and above_h and is_bull and not vpa_block_long
            short_entry = sl_val < -slope_thresh and below_l and is_bear and not vpa_block_short

            cooldown_ok = (j - last_exit_bar) >= cooldown

            # ── ENTRY ──
            if pos == 0 and cooldown_ok:
                if long_entry:
                    pos = 1
                    entry_price = float(cl)
                    sl_price = entry_price - sl_atr_mult * atr_arr[j]
                    entry_bar = j
                    entry_slope = sl_val
                    entry_rel_vol = rel_v
                    entry_score = (1
                        + (1 if rel_v >= 1.0 else 0)
                        + (1 if rel_v >= vol_thresh_high else 0)
                        + (1 if spr else 0)
                        + (1 if cp > 0.6 else 0))
                elif short_entry:
                    pos = -1
                    entry_price = float(cl)
                    sl_price = entry_price + sl_atr_mult * atr_arr[j]
                    entry_bar = j
                    entry_slope = sl_val
                    entry_rel_vol = rel_v
                    entry_score = (1
                        + (1 if rel_v >= 1.0 else 0)
                        + (1 if rel_v >= vol_thresh_high else 0)
                        + (1 if ut else 0)
                        + (1 if cp < 0.4 else 0))

            # ── EXIT ──
            if pos == 1:
                do_exit = False
                if lo <= sl_price:
                    do_exit = True
                elif sl_val < -exit_slope_rev:
                    do_exit = True
                elif cl < el_j:
                    do_exit = True
                elif (j - entry_bar) >= max_hold:
                    do_exit = True
                if do_exit:
                    last_exit_bar = j
                    pos = 0

            elif pos == -1:
                do_exit = False
                if hi >= sl_price:
                    do_exit = True
                elif sl_val > exit_slope_rev:
                    do_exit = True
                elif cl > eh:
                    do_exit = True
                elif (j - entry_bar) >= max_hold:
                    do_exit = True
                if do_exit:
                    last_exit_bar = j
                    pos = 0

        # After walking all bars, check final state
        is_fno = fno_set and sym in fno_set
        if pos == 1:
            bars_held = n - 1 - entry_bar
            live_pnl = (c[-1] - entry_price) / entry_price * 100
            entry_date = df.index[entry_bar].strftime('%Y-%m-%d')
            longs.append({
                "Symbol": sym,
                "F&O": "Yes" if is_fno else "",
                "Close": round(float(c[-1]), 2),
                "Entry": round(float(entry_price), 2),
                "Slope %": round(float(entry_slope), 2),
                "Rel Vol": round(float(entry_rel_vol), 2),
                "VPA Score": entry_score,
                "Stars": "★" * entry_score,
                "PnL %": round(float(live_pnl), 2),
                "Bars": bars_held,
                "Entry Date": entry_date,
            })
        elif pos == -1 and is_fno:
            # Short signals only for F&O stocks
            bars_held = n - 1 - entry_bar
            live_pnl = (entry_price - c[-1]) / entry_price * 100
            entry_date = df.index[entry_bar].strftime('%Y-%m-%d')
            shorts.append({
                "Symbol": sym,
                "Close": round(float(c[-1]), 2),
                "Entry": round(float(entry_price), 2),
                "Slope %": round(float(entry_slope), 2),
                "Rel Vol": round(float(entry_rel_vol), 2),
                "VPA Score": entry_score,
                "Stars": "★" * entry_score,
                "PnL %": round(float(live_pnl), 2),
                "Bars": bars_held,
                "Entry Date": entry_date,
            })

    longs_df = pd.DataFrame(longs).sort_values("VPA Score", ascending=False) if longs else pd.DataFrame()
    shorts_df = pd.DataFrame(shorts).sort_values("VPA Score", ascending=False) if shorts else pd.DataFrame()
    return longs_df, shorts_df


# ─── Orchestrator: batch download + compute, cache only results ──
@st.cache_data(ttl=600, show_spinner=False)
def scan_all(symbols_tuple, interval, fno_tuple, chunk_size=50):
    symbols = list(symbols_tuple)
    fno_set = set(fno_tuple)
    all_longs = []
    all_shorts = []
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        try:
            data = download_chunk(chunk, interval)
        except Exception:
            continue
        longs_df, shorts_df = compute_signals(data, chunk, interval, fno_set)
        if not longs_df.empty:
            all_longs.append(longs_df)
        if not shorts_df.empty:
            all_shorts.append(shorts_df)
        del data
    longs = (pd.concat(all_longs, ignore_index=True)
             .sort_values("VPA Score", ascending=False)) if all_longs else pd.DataFrame()
    shorts = (pd.concat(all_shorts, ignore_index=True)
              .sort_values("VPA Score", ascending=False)) if all_shorts else pd.DataFrame()
    return longs, shorts


# ─── UI ──────────────────────────────────────────────────────────
st.title("5-Star Stocks — EMA34 Slope Scanner")
st.caption("EMA34 Band + EMA5 Slope(3-bar) + VPA | Nifty 500")

with st.sidebar:
    timeframe = st.radio("Timeframe", ["Daily", "Weekly"], index=0)
    if st.button("Refresh Data"):
        st.cache_data.clear()

interval = "1d" if timeframe == "Daily" else "1wk"

try:
    symbols = fetch_nifty500()
except Exception as e:
    st.error(f"Failed to load Nifty 500 list: {e}")
    st.stop()

try:
    fno_set = fetch_fno_stocks()
except Exception:
    fno_set = set()
    st.sidebar.warning("F&O list unavailable")

st.sidebar.metric("Stocks", len(symbols))
st.sidebar.metric("F&O Stocks", len(fno_set))

status = st.empty()
status.info(f"Scanning {len(symbols)} stocks ({timeframe.lower()})... "
            "this takes ~2-3 min on first run, cached for 10 min after.")
longs_df, shorts_df = scan_all(tuple(symbols), interval, tuple(sorted(fno_set)))
status.empty()

st.sidebar.markdown("---")
st.sidebar.metric("Long Signals", len(longs_df))
st.sidebar.metric("Short Signals", len(shorts_df))
st.sidebar.caption(f"Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Long Signals ({len(longs_df)})", divider="green")
    if longs_df.empty:
        st.info("No long signals found.")
    else:
        st.dataframe(longs_df.reset_index(drop=True), use_container_width=True, hide_index=True)

with col2:
    st.subheader(f"Short Signals ({len(shorts_df)})", divider="red")
    st.caption("F&O stocks only")
    if shorts_df.empty:
        st.info("No short signals found.")
    else:
        st.dataframe(shorts_df.reset_index(drop=True), use_container_width=True, hide_index=True)
