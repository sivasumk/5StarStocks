import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="Nifty 100 Momentum Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Weight configuration for momentum score ──────────────────────
# These weights were chosen to balance trend-following, momentum
# oscillators, volume confirmation, and relative performance.
WEIGHTS = {
    "ema":      0.20,   # EMA34 position + EMA5 slope (primary trend)
    "rs":       0.20,   # Relative strength vs index (stock selection)
    "rsi":      0.15,   # RSI momentum
    "smi":      0.15,   # SMI momentum timing
    "adx":      0.15,   # ADX trend strength + direction
    "obv":      0.10,   # OBV volume confirmation
    "atr":      0.05,   # ATR expansion/contraction energy
}

TREND_DAYS = 7   # number of trading days for trend computation


# ═══════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400)
def fetch_nifty100_symbols():
    """Load Nifty 100 constituents.

    Tries the live niftyindices.com CSV first; falls back to the bundled
    static CSV if the fetch fails (needed on hosts that block outbound
    requests to niftyindices.com, e.g. Streamlit Community Cloud).
    """
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/csv,text/plain,*/*",
        "Referer": "https://www.niftyindices.com/",
    }
    source = "static CSV"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        source = "niftyindices.com (live)"
    except Exception:
        df = pd.read_csv(BASE_DIR / "nifty100.csv")

    sym_col = [c for c in df.columns if "symbol" in c.lower()][0]
    symbols = df[sym_col].str.strip().tolist()
    return sorted(symbols), source


@st.cache_data(ttl=300, show_spinner=False)
def download_ohlcv(symbols):
    """Download 6 months of daily OHLCV for given symbols."""
    tickers = [s + ".NS" for s in symbols]
    data = yf.download(
        tickers, period="6mo", interval="1d",
        group_by="ticker", threads=True, progress=False,
    )
    return data


@st.cache_data(ttl=300, show_spinner=False)
def download_index():
    """Download Nifty 100 index for relative-strength computation."""
    idx = yf.download("^CNX100", period="6mo", interval="1d", progress=False)
    return idx


# ═══════════════════════════════════════════════════════════════════
#  INDICATOR COMPUTATIONS
# ═══════════════════════════════════════════════════════════════════

def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def compute_smi(high, low, close, k_len=10, d_len=3, ema_len=5):
    """Stochastic Momentum Index — matches the TradingView Pine script."""
    hh = high.rolling(k_len).max()
    ll = low.rolling(k_len).min()
    hlr = hh - ll                           # highestLowestRange
    rr = close - (hh + ll) / 2              # relativeRange
    num = _ema(_ema(rr, d_len), d_len)
    den = _ema(_ema(hlr, d_len), d_len)
    smi = 200 * (num / den.replace(0, np.nan))
    smi_signal = _ema(smi, ema_len)
    return smi, smi_signal


def compute_obv(close, volume):
    direction = np.sign(close.diff())
    return (volume * direction).cumsum()


def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di


# ═══════════════════════════════════════════════════════════════════
#  SCORING FUNCTIONS  (each returns a value in [-100, +100])
# ═══════════════════════════════════════════════════════════════════

def _clamp(x, lo=-100, hi=100):
    return max(lo, min(hi, x))


def score_rsi(rsi_val):
    """RSI 50 → 0,  60 → +50,  70+ → +100,  40 → −50,  30− → −100."""
    return _clamp((rsi_val - 50) * 5)


def score_smi(smi_val):
    """SMI ±40 → ±100."""
    return _clamp(smi_val * 2.5)


def score_adx(adx_val, plus_di, minus_di):
    """Direction from DI, strength from ADX."""
    direction = 1 if plus_di > minus_di else -1
    strength = _clamp((adx_val - 15) / 30 * 100, 0, 100)
    return direction * strength


def score_ema(close_val, ema34_val, ema5_slope_pct):
    """Price vs EMA34 position + EMA5 slope."""
    if ema34_val == 0:
        return 0
    pos = _clamp((close_val - ema34_val) / ema34_val * 1000)
    slope = _clamp(ema5_slope_pct * 40)
    return 0.5 * pos + 0.5 * slope


def score_rs(stock_ret, index_ret):
    """Relative strength: outperformance in pct points → score."""
    diff = stock_ret - index_ret
    return _clamp(diff * 8)


def score_obv(obv_series, avg_vol):
    """OBV 7-day slope normalised by average volume."""
    if len(obv_series) < 8 or avg_vol == 0:
        return 0
    slope = (obv_series.iloc[-1] - obv_series.iloc[-8]) / (avg_vol * 7) * 100
    return _clamp(slope * 15)


def score_atr(atr_ratio, price_direction):
    """ATR expansion in trend direction = momentum."""
    expansion = (atr_ratio - 1) * 100
    return _clamp(expansion * price_direction * 3)


# ═══════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS — processes all stocks
# ═══════════════════════════════════════════════════════════════════

def analyse_all(data, symbols, index_data):
    """Compute indicators, scores and 7-day trends for every stock."""

    # Index return for relative strength
    if index_data is not None and "Close" in index_data.columns and len(index_data) >= 21:
        idx_close = index_data["Close"].dropna()
        if hasattr(idx_close, "columns"):
            idx_close = idx_close.iloc[:, 0]
        idx_ret_20 = (idx_close.iloc[-1] / idx_close.iloc[-21] - 1) * 100 if len(idx_close) >= 21 else 0
    else:
        idx_ret_20 = 0

    rows = []
    skipped = []
    for sym in symbols:
        ticker = sym + ".NS"
        try:
            if ticker in data.columns.get_level_values(0):
                df = data[ticker].dropna(subset=["Close"]).copy()
            else:
                skipped.append((sym, "No data from yfinance"))
                continue
        except Exception as exc:
            skipped.append((sym, f"Error: {exc}"))
            continue

        if len(df) < 35:
            skipped.append((sym, f"Only {len(df)} bars (need 35)"))
            continue

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        vol = df["Volume"]

        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        if hasattr(high, "columns"):
            high = high.iloc[:, 0]
        if hasattr(low, "columns"):
            low = low.iloc[:, 0]
        if hasattr(vol, "columns"):
            vol = vol.iloc[:, 0]

        # ── Compute indicators on full series ──
        rsi = compute_rsi(close)
        smi, smi_sig = compute_smi(high, low, close)
        obv = compute_obv(close, vol)
        atr = compute_atr(high, low, close)
        adx, plus_di, minus_di = compute_adx(high, low, close)
        ema34 = _ema(close, 34)
        ema5 = _ema(close, 5)

        # EMA5 slope (3-bar pct change)
        ema5_slope = (ema5 / ema5.shift(3) - 1) * 100

        # ATR ratio (current / 20-day SMA of ATR)
        atr_avg20 = atr.rolling(20).mean()
        atr_ratio = atr / atr_avg20.replace(0, np.nan)

        # OBV average volume for normalisation
        avg_vol_20 = vol.rolling(20).mean()

        # 20-day stock return for relative strength
        stock_ret_20 = (close / close.shift(20) - 1) * 100

        # ── Current values (last bar) ──
        cur = {
            "rsi": rsi.iloc[-1],
            "smi": smi.iloc[-1],
            "smi_sig": smi_sig.iloc[-1],
            "adx": adx.iloc[-1],
            "plus_di": plus_di.iloc[-1],
            "minus_di": minus_di.iloc[-1],
            "ema34": ema34.iloc[-1],
            "ema5": ema5.iloc[-1],
            "ema5_slope": ema5_slope.iloc[-1],
            "atr": atr.iloc[-1],
            "atr_ratio": atr_ratio.iloc[-1],
            "obv": obv.iloc[-1],
            "avg_vol": avg_vol_20.iloc[-1],
            "stock_ret": stock_ret_20.iloc[-1],
            "close": close.iloc[-1],
        }

        # Skip if any key value is NaN
        if any(pd.isna(v) for v in [cur["rsi"], cur["smi"], cur["adx"], cur["ema34"]]):
            skipped.append((sym, "NaN in key indicators"))
            continue

        # ── Component scores ──
        s_rsi = score_rsi(cur["rsi"])
        s_smi = score_smi(cur["smi"])
        s_adx = score_adx(cur["adx"], cur["plus_di"], cur["minus_di"])
        s_ema = score_ema(cur["close"], cur["ema34"], cur["ema5_slope"])
        s_rs  = score_rs(cur["stock_ret"], idx_ret_20)
        s_obv = score_obv(obv, cur["avg_vol"])
        price_dir = 1 if cur["close"] > ema34.iloc[-1] else -1
        s_atr = score_atr(cur["atr_ratio"], price_dir)

        momentum = (
            WEIGHTS["rsi"] * s_rsi
            + WEIGHTS["smi"] * s_smi
            + WEIGHTS["adx"] * s_adx
            + WEIGHTS["ema"] * s_ema
            + WEIGHTS["rs"]  * s_rs
            + WEIGHTS["obv"] * s_obv
            + WEIGHTS["atr"] * s_atr
        )

        # ── 7-day trend: compute momentum score for each of last 7 bars ──
        trend_scores = []
        trend_rsi = []
        trend_smi_vals = []
        trend_adx_vals = []
        trend_ema_scores = []
        trend_rs = []
        trend_obv_scores = []

        for offset in range(TREND_DAYS, 0, -1):
            idx = -1 - offset
            if abs(idx) >= len(close):
                continue
            try:
                t_rsi = rsi.iloc[idx]
                t_smi = smi.iloc[idx]
                t_adx_v = adx.iloc[idx]
                t_pdi = plus_di.iloc[idx]
                t_mdi = minus_di.iloc[idx]
                t_ema34 = ema34.iloc[idx]
                t_ema5s = ema5_slope.iloc[idx]
                t_cl = close.iloc[idx]
                t_atr_r = atr_ratio.iloc[idx]
                t_avg_vol = avg_vol_20.iloc[idx]

                # Stock return 20 days back from that point
                ret_idx = idx - 20
                if abs(ret_idx) < len(close):
                    t_sret = (close.iloc[idx] / close.iloc[ret_idx] - 1) * 100
                else:
                    t_sret = 0

                ts_rsi = score_rsi(t_rsi)
                ts_smi = score_smi(t_smi)
                ts_adx = score_adx(t_adx_v, t_pdi, t_mdi)
                ts_ema = score_ema(t_cl, t_ema34, t_ema5s)
                ts_rs  = score_rs(t_sret, idx_ret_20)

                # OBV score at that point
                obv_slice = obv.iloc[:idx+1] if idx != -1 else obv
                ts_obv = score_obv(obv_slice, t_avg_vol)

                t_pdir = 1 if t_cl > t_ema34 else -1
                ts_atr = score_atr(t_atr_r, t_pdir)

                t_mom = (
                    WEIGHTS["rsi"] * ts_rsi + WEIGHTS["smi"] * ts_smi
                    + WEIGHTS["adx"] * ts_adx + WEIGHTS["ema"] * ts_ema
                    + WEIGHTS["rs"] * ts_rs + WEIGHTS["obv"] * ts_obv
                    + WEIGHTS["atr"] * ts_atr
                )
                trend_scores.append(round(t_mom, 1))
                trend_rsi.append(round(t_rsi, 1))
                trend_smi_vals.append(round(t_smi, 1))
                trend_adx_vals.append(round(t_adx_v, 1))
                trend_ema_scores.append(round(ts_ema, 1))
                trend_rs.append(round(ts_rs, 1))
                trend_obv_scores.append(round(ts_obv, 1))
            except (IndexError, KeyError):
                continue

        # Append current values to trend arrays
        trend_scores.append(round(momentum, 1))
        trend_rsi.append(round(cur["rsi"], 1))
        trend_smi_vals.append(round(cur["smi"], 1))
        trend_adx_vals.append(round(cur["adx"], 1))
        trend_ema_scores.append(round(s_ema, 1))
        trend_rs.append(round(s_rs, 1))
        trend_obv_scores.append(round(s_obv, 1))

        # Trend direction helper
        def trend_dir(arr):
            if len(arr) < 2:
                return "→"
            diff = arr[-1] - arr[0]
            if diff > 3:
                return "↑"
            elif diff < -3:
                return "↓"
            return "→"

        # RSI zone label
        rsi_val = cur["rsi"]
        if rsi_val >= 70:
            rsi_zone = "Strong Bull"
        elif rsi_val >= 60:
            rsi_zone = "Bullish"
        elif rsi_val >= 40:
            rsi_zone = "Neutral"
        elif rsi_val >= 30:
            rsi_zone = "Bearish"
        else:
            rsi_zone = "Strong Bear"

        # SMI zone
        smi_val = cur["smi"]
        if smi_val >= 40:
            smi_zone = "Overbought"
        elif smi_val >= 10:
            smi_zone = "Bullish"
        elif smi_val >= -10:
            smi_zone = "Neutral"
        elif smi_val >= -40:
            smi_zone = "Bearish"
        else:
            smi_zone = "Oversold"

        # ATR state
        atr_r = cur["atr_ratio"]
        if atr_r >= 1.3:
            atr_state = "Expanding"
        elif atr_r >= 1.0:
            atr_state = "Slight Exp"
        elif atr_r >= 0.8:
            atr_state = "Slight Con"
        else:
            atr_state = "Contracting"

        # ADX trend strength
        adx_val = cur["adx"]
        if adx_val >= 40:
            adx_str = "Strong"
        elif adx_val >= 25:
            adx_str = "Trending"
        elif adx_val >= 15:
            adx_str = "Weak"
        else:
            adx_str = "No Trend"

        # EMA status
        above_ema = cur["close"] > cur["ema34"]
        ema_status = "Above" if above_ema else "Below"

        rows.append({
            "Symbol": sym,
            "Close": round(float(cur["close"]), 2),
            "Score": round(momentum, 1),
            "Score Trend": trend_scores,
            "Score Dir": trend_dir(trend_scores),

            "RSI": round(rsi_val, 1),
            "RSI Zone": rsi_zone,
            "RSI Trend": trend_rsi,
            "RSI Dir": trend_dir(trend_rsi),

            "SMI": round(smi_val, 1),
            "SMI Sig": round(cur["smi_sig"], 1),
            "SMI Zone": smi_zone,
            "SMI Trend": trend_smi_vals,
            "SMI Dir": trend_dir(trend_smi_vals),

            "OBV Score": round(s_obv, 1),
            "OBV Trend": trend_obv_scores,
            "OBV Dir": trend_dir(trend_obv_scores),

            "ATR": round(float(cur["atr"]), 2),
            "ATR Ratio": round(float(atr_r), 2),
            "ATR State": atr_state,

            "ADX": round(adx_val, 1),
            "+DI": round(float(cur["plus_di"]), 1),
            "-DI": round(float(cur["minus_di"]), 1),
            "ADX Str": adx_str,
            "ADX Trend": trend_adx_vals,
            "ADX Dir": trend_dir(trend_adx_vals),

            "RS Score": round(s_rs, 1),
            "RS Trend": trend_rs,
            "RS Dir": trend_dir(trend_rs),

            "EMA 34": round(float(cur["ema34"]), 2),
            "EMA 5 Slope%": round(float(cur["ema5_slope"]), 3),
            "EMA Status": ema_status,
            "EMA Score": round(s_ema, 1),
            "EMA Trend": trend_ema_scores,
            "EMA Dir": trend_dir(trend_ema_scores),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("Score", ascending=False).reset_index(drop=True)
        result.index = result.index + 1
        result.index.name = "Rank"
    if skipped:
        print(f"SKIPPED STOCKS: {skipped}")
    return result, skipped


# ═══════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════

def color_score(val):
    """Color-code momentum score cells."""
    if isinstance(val, (int, float)):
        if val >= 50:
            return "background-color: #0d6e3a; color: white"
        elif val >= 20:
            return "background-color: #198754; color: white"
        elif val >= 0:
            return "background-color: #d1e7dd; color: black"
        elif val >= -20:
            return "background-color: #f8d7da; color: black"
        elif val >= -50:
            return "background-color: #dc3545; color: white"
        else:
            return "background-color: #8b0000; color: white"
    return ""


def color_rsi(val):
    if isinstance(val, (int, float)):
        if val >= 70:
            return "background-color: #0d6e3a; color: white"
        elif val >= 60:
            return "background-color: #198754; color: white"
        elif val >= 40:
            return "background-color: #6c757d; color: white"
        elif val >= 30:
            return "background-color: #dc3545; color: white"
        else:
            return "background-color: #8b0000; color: white"
    return ""


def color_trend_dir(val):
    if val == "↑":
        return "color: #00cc44; font-weight: bold; font-size: 1.2em"
    elif val == "↓":
        return "color: #ff3333; font-weight: bold; font-size: 1.2em"
    return "color: #888888"


def main():
    st.title("Nifty 100 — Momentum Ranking Dashboard")
    st.caption("RSI | SMI | OBV | ATR | ADX | Relative Strength | EMA 34 + EMA 5 Slope")

    # ── Sidebar ──
    with st.sidebar:
        st.header("Controls")
        if st.button("Refresh Data", width="stretch"):
            st.cache_data.clear()

        st.markdown("---")
        st.subheader("Filters")
        zone_filter = st.multiselect(
            "RSI Zone",
            ["Strong Bull", "Bullish", "Neutral", "Bearish", "Strong Bear"],
            default=[],
        )
        ema_filter = st.multiselect(
            "EMA Status",
            ["Above", "Below"],
            default=[],
        )
        atr_filter = st.multiselect(
            "ATR State",
            ["Expanding", "Slight Exp", "Slight Con", "Contracting"],
            default=[],
        )
        adx_filter = st.multiselect(
            "ADX Strength",
            ["Strong", "Trending", "Weak", "No Trend"],
            default=[],
        )
        score_range = st.slider("Momentum Score Range", -100, 100, (-100, 100))

        st.markdown("---")
        st.subheader("Score Weights")
        st.caption("Current weights (edit in code)")
        for k, v in WEIGHTS.items():
            st.text(f"{k.upper():>5s}: {v:.0%}")

    # ── Fetch data ──
    with st.spinner("Loading Nifty 100 list..."):
        try:
            symbols, sym_source = fetch_nifty100_symbols()
        except Exception as e:
            st.error(f"Failed to load Nifty 100 list: {e}")
            st.stop()

    st.sidebar.metric("Stocks in Index", len(symbols))
    st.sidebar.caption(f"List source: {sym_source}")

    with st.spinner(f"Downloading daily data for {len(symbols)} stocks..."):
        data = download_ohlcv(symbols)

    with st.spinner("Downloading Nifty 100 index data..."):
        try:
            index_data = download_index()
        except Exception:
            index_data = None
            st.sidebar.warning("Index data unavailable — RS will use 0 baseline")

    with st.spinner("Computing indicators & ranking..."):
        result, skipped = analyse_all(data, symbols, index_data)

    if result.empty:
        st.error("No data could be processed. Try refreshing.")
        st.stop()

    if skipped:
        with st.sidebar.expander(f"Skipped Stocks ({len(skipped)})"):
            for sym, reason in skipped:
                st.text(f"{sym}: {reason}")

    # ── Apply filters ──
    filtered = result.copy()
    if zone_filter:
        filtered = filtered[filtered["RSI Zone"].isin(zone_filter)]
    if ema_filter:
        filtered = filtered[filtered["EMA Status"].isin(ema_filter)]
    if atr_filter:
        filtered = filtered[filtered["ATR State"].isin(atr_filter)]
    if adx_filter:
        filtered = filtered[filtered["ADX Str"].isin(adx_filter)]
    filtered = filtered[
        (filtered["Score"] >= score_range[0]) & (filtered["Score"] <= score_range[1])
    ]

    # ── Summary metrics ──
    st.sidebar.markdown("---")
    n_bull = len(result[result["Score"] > 20])
    n_bear = len(result[result["Score"] < -20])
    n_neutral = len(result) - n_bull - n_bear
    st.sidebar.metric("Bullish (>20)", n_bull)
    st.sidebar.metric("Neutral", n_neutral)
    st.sidebar.metric("Bearish (<-20)", n_bear)
    avg_score = result["Score"].mean()
    st.sidebar.metric("Avg Score", f"{avg_score:.1f}")
    st.sidebar.caption(f"Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ── Market breadth bar ──
    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric("Bullish Stocks", f"{n_bull} / {len(result)}", delta=f"{n_bull/len(result)*100:.0f}%")
    col_b2.metric("Average Momentum", f"{avg_score:.1f}", delta="Bullish" if avg_score > 10 else ("Bearish" if avg_score < -10 else "Neutral"))
    col_b3.metric("Showing", f"{len(filtered)} / {len(result)}", delta="filtered" if len(filtered) < len(result) else "all")

    st.markdown("---")

    # ── Display mode ──
    view = st.radio("View", ["Full Dashboard", "Compact Ranking"], horizontal=True)

    if view == "Compact Ranking":
        # Compact: just symbol, score, key indicators, trend arrows
        compact_cols = [
            "Symbol", "Close", "Score", "Score Dir",
            "RSI", "RSI Zone", "RSI Dir",
            "SMI", "SMI Zone", "SMI Dir",
            "ADX", "ADX Str", "ADX Dir",
            "EMA Status", "EMA 5 Slope%", "EMA Dir",
            "RS Score", "RS Dir",
            "ATR State",
            "OBV Score", "OBV Dir",
        ]
        display_df = filtered[compact_cols].copy()

        styled = display_df.style.map(color_score, subset=["Score"]) \
            .map(color_rsi, subset=["RSI"]) \
            .map(color_trend_dir, subset=["Score Dir", "RSI Dir", "SMI Dir", "ADX Dir", "EMA Dir", "RS Dir", "OBV Dir"])

        st.dataframe(styled, width="stretch", height=800)

    else:
        # Full dashboard with sparkline trends
        st.subheader("Full Momentum Dashboard")

        # Configure columns for st.dataframe with sparkline charts
        display_cols = [
            "Symbol", "Close", "Score", "Score Trend", "Score Dir",
            "RSI", "RSI Zone", "RSI Trend", "RSI Dir",
            "SMI", "SMI Sig", "SMI Zone", "SMI Trend", "SMI Dir",
            "ADX", "+DI", "-DI", "ADX Str", "ADX Trend", "ADX Dir",
            "EMA 34", "EMA Status", "EMA 5 Slope%", "EMA Score", "EMA Trend", "EMA Dir",
            "RS Score", "RS Trend", "RS Dir",
            "OBV Score", "OBV Trend", "OBV Dir",
            "ATR", "ATR Ratio", "ATR State",
        ]

        display_df = filtered[display_cols]

        col_config = {
            "Score Trend": st.column_config.LineChartColumn(
                "Score 7D", width="small", y_min=-100, y_max=100,
            ),
            "RSI Trend": st.column_config.LineChartColumn(
                "RSI 7D", width="small", y_min=0, y_max=100,
            ),
            "SMI Trend": st.column_config.LineChartColumn(
                "SMI 7D", width="small", y_min=-80, y_max=80,
            ),
            "ADX Trend": st.column_config.LineChartColumn(
                "ADX 7D", width="small", y_min=0, y_max=60,
            ),
            "EMA Trend": st.column_config.LineChartColumn(
                "EMA 7D", width="small", y_min=-100, y_max=100,
            ),
            "RS Trend": st.column_config.LineChartColumn(
                "RS 7D", width="small", y_min=-100, y_max=100,
            ),
            "OBV Trend": st.column_config.LineChartColumn(
                "OBV 7D", width="small", y_min=-100, y_max=100,
            ),
            "Score": st.column_config.NumberColumn(format="%.1f"),
            "RSI": st.column_config.NumberColumn(format="%.1f"),
            "SMI": st.column_config.NumberColumn(format="%.1f"),
            "SMI Sig": st.column_config.NumberColumn("SMI Signal", format="%.1f"),
            "ADX": st.column_config.NumberColumn(format="%.1f"),
            "+DI": st.column_config.NumberColumn(format="%.1f"),
            "-DI": st.column_config.NumberColumn(format="%.1f"),
            "EMA 34": st.column_config.NumberColumn(format="%.2f"),
            "EMA 5 Slope%": st.column_config.NumberColumn("EMA5 Slope%", format="%.3f"),
            "EMA Score": st.column_config.NumberColumn(format="%.1f"),
            "RS Score": st.column_config.NumberColumn(format="%.1f"),
            "OBV Score": st.column_config.NumberColumn(format="%.1f"),
            "ATR": st.column_config.NumberColumn(format="%.2f"),
            "ATR Ratio": st.column_config.NumberColumn(format="%.2f"),
        }

        st.dataframe(
            display_df,
            column_config=col_config,
            width="stretch",
            height=900,
        )

    # ── Top / Bottom 10 ──
    st.markdown("---")
    col_t, col_bo = st.columns(2)
    with col_t:
        st.subheader("Top 10 Momentum", divider="green")
        top10 = result.head(10)[["Symbol", "Close", "Score", "RSI Zone", "SMI Zone", "ADX Str", "EMA Status", "ATR State"]].copy()
        st.dataframe(top10, width="stretch", hide_index=True)

    with col_bo:
        st.subheader("Bottom 10 Momentum", divider="red")
        bot10 = result.tail(10)[["Symbol", "Close", "Score", "RSI Zone", "SMI Zone", "ADX Str", "EMA Status", "ATR State"]].copy()
        st.dataframe(bot10, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
