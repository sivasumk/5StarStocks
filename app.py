import streamlit as st
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="5-Star Stocks Scanner", layout="wide")

BASE_DIR = Path(__file__).parent
IST = timezone(timedelta(hours=5, minutes=30))  # no DST, so a fixed offset is exact

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

    fno = fno_set or set()
    longs = []
    shorts = []
    trades = []   # every closed trade in the downloaded history

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

            # ── EXIT ── (from the bar after entry; the entry bar's low/high
            # printed before the close we entered at)
            if pos != 0 and j > entry_bar:
                do_exit = False
                reason = ""
                exit_price = cl
                if pos == 1:
                    if lo <= sl_price:
                        do_exit = True; reason = "Stop Loss"
                        exit_price = min(op_j, sl_price)  # gap below stop fills at open
                    elif sl_val < -exit_slope_rev:
                        do_exit = True; reason = "Slope Flip"
                    elif cl < el_j:
                        do_exit = True; reason = "Below Band"
                    elif (j - entry_bar) >= max_hold:
                        do_exit = True; reason = "Max Hold"
                else:
                    if hi >= sl_price:
                        do_exit = True; reason = "Stop Loss"
                        exit_price = max(op_j, sl_price)  # gap above stop fills at open
                    elif sl_val > exit_slope_rev:
                        do_exit = True; reason = "Slope Flip"
                    elif cl > eh:
                        do_exit = True; reason = "Above Band"
                    elif (j - entry_bar) >= max_hold:
                        do_exit = True; reason = "Max Hold"
                if do_exit:
                    exit_pnl = pos * (exit_price - entry_price) / entry_price * 100
                    trades.append({
                        "Symbol": sym,
                        "Side": "Long" if pos == 1 else "Short",
                        "F&O": "Yes" if sym in fno else "",
                        "Reason": reason,
                        "Entry": round(float(entry_price), 2),
                        "Exit": round(float(exit_price), 2),
                        "PnL %": round(float(exit_pnl), 2),
                        "Bars Held": j - entry_bar,
                        "VPA Score": entry_score,
                        "Entry Date": df.index[entry_bar].strftime('%Y-%m-%d'),
                        "Exit Date": df.index[j].strftime('%Y-%m-%d'),
                        "_latest": j == n - 1,
                    })
                    last_exit_bar = j
                    pos = 0

        # After walking all bars, check final state
        is_fno = sym in fno
        if pos == 1:
            bars_held = n - 1 - entry_bar
            live_pnl = (c[-1] - entry_price) / entry_price * 100
            entry_date = df.index[entry_bar].strftime('%Y-%m-%d')
            longs.append({
                "Symbol": sym,
                "F&O": "Yes" if is_fno else "",
                "Close": round(float(c[-1]), 2),
                "Entry": round(float(entry_price), 2),
                "Stop": round(float(sl_price), 2),
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
                "Stop": round(float(sl_price), 2),
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
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        # Shorts are only traded in F&O stocks
        trades_df = trades_df[(trades_df["Side"] == "Long") | (trades_df["F&O"] == "Yes")]
    return longs_df, shorts_df, trades_df


# ─── Orchestrator: batch download + compute, cache only results ──
@st.cache_data(ttl=600, show_spinner=False)
def scan_all(symbols_tuple, interval, fno_tuple, chunk_size=50):
    symbols = list(symbols_tuple)
    fno_set = set(fno_tuple)
    all_longs = []
    all_shorts = []
    all_trades = []
    failed_chunks = 0
    empty_chunks = 0
    processed = 0
    missing = []
    last_bar = None
    errors = []
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        try:
            data = download_chunk(chunk, interval)
        except Exception as e:
            failed_chunks += 1
            errors.append(f"Chunk {i//chunk_size + 1} download failed: {e}")
            continue
        if data is None or data.empty:
            empty_chunks += 1
            continue
        longs_df, shorts_df, trades_df = compute_signals(data, chunk, interval, fno_set)
        present = set(data.columns.get_level_values(0))
        for sym in chunk:
            if sym + ".NS" in present and data[sym + ".NS"]["Close"].notna().any():
                processed += 1
            else:
                missing.append(sym)
        chunk_last = data.index.max()
        last_bar = chunk_last if last_bar is None else max(last_bar, chunk_last)
        if not longs_df.empty:
            all_longs.append(longs_df)
        if not shorts_df.empty:
            all_shorts.append(shorts_df)
        if not trades_df.empty:
            all_trades.append(trades_df)
        del data
    longs = (pd.concat(all_longs, ignore_index=True)
             .sort_values("VPA Score", ascending=False)) if all_longs else pd.DataFrame()
    shorts = (pd.concat(all_shorts, ignore_index=True)
              .sort_values("VPA Score", ascending=False)) if all_shorts else pd.DataFrame()
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    diagnostics = {
        "processed": processed,
        "failed_chunks": failed_chunks,
        "empty_chunks": empty_chunks,
        "errors": errors[:3],
        "missing": missing,
        "scan_time": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        "last_bar": last_bar.strftime("%Y-%m-%d") if last_bar is not None else None,
    }
    return longs, shorts, trades, diagnostics


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
longs_df, shorts_df, trades_df, diag = scan_all(tuple(symbols), interval, tuple(sorted(fno_set)))
status.empty()

EXIT_COLS = ["Symbol", "Side", "Reason", "Entry", "Exit", "PnL %", "Bars Held", "Entry Date"]
if trades_df.empty:
    exited_df = pd.DataFrame()
else:
    exited_df = (trades_df[trades_df["_latest"]][EXIT_COLS]
                 .sort_values("PnL %", ascending=False))

# Show diagnostics if anything went wrong
if diag["failed_chunks"] or diag["empty_chunks"] or diag["missing"]:
    with st.expander(f"⚠️ Scan diagnostics ({diag['processed']}/{len(symbols)} processed)"):
        st.write(f"Failed chunks: {diag['failed_chunks']}, Empty chunks: {diag['empty_chunks']}")
        for err in diag["errors"]:
            st.code(err)
        if diag["missing"]:
            st.write(f"No data for {len(diag['missing'])} symbols: "
                     + ", ".join(diag["missing"][:50])
                     + (" …" if len(diag["missing"]) > 50 else ""))

st.sidebar.markdown("---")
st.sidebar.metric("Long Signals", len(longs_df))
st.sidebar.metric("Short Signals", len(shorts_df))
st.sidebar.metric("Exited on Latest Bar", len(exited_df))
st.sidebar.caption(f"Last scan: {diag['scan_time']}")
st.sidebar.caption(f"Latest bar: {diag['last_bar']}")

# The latest bar is still forming during market hours (and all week on the
# weekly timeframe), so its entries/exits can change before it closes.
now_ist = datetime.now(IST)
market_open = (now_ist.weekday() < 5
               and (9, 15) <= (now_ist.hour, now_ist.minute) < (15, 30))
bar_forming = diag["last_bar"] is not None and (
    (interval == "1d" and market_open and diag["last_bar"] == now_ist.strftime("%Y-%m-%d"))
    or (interval == "1wk" and now_ist.weekday() < 5
        and now_ist.date() - timedelta(days=now_ist.weekday()) <= datetime.strptime(diag["last_bar"], "%Y-%m-%d").date()))
if bar_forming:
    st.warning(f"The latest {'daily' if interval == '1d' else 'weekly'} bar "
               f"({diag['last_bar']}) is still forming — signals and exits on it "
               "may change before it closes.")

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Long Signals ({len(longs_df)})", divider="green")
    if longs_df.empty:
        st.info("No long signals found.")
    else:
        st.dataframe(longs_df.reset_index(drop=True), width="stretch", hide_index=True)

with col2:
    st.subheader(f"Short Signals ({len(shorts_df)})", divider="red")
    st.caption("F&O stocks only")
    if shorts_df.empty:
        st.info("No short signals found.")
    else:
        st.dataframe(shorts_df.reset_index(drop=True), width="stretch", hide_index=True)

st.subheader(f"Exited on Latest Bar ({len(exited_df)})", divider="orange")
st.caption(f"Positions that closed on the {diag['last_bar']} "
           f"{'daily' if interval == '1d' else 'weekly'} bar. "
           "Stop-loss exits are priced at the stop (or the open on a gap).")
if exited_df.empty:
    st.info("No exits on the latest bar.")
else:
    st.dataframe(exited_df.reset_index(drop=True), width="stretch", hide_index=True)


# ─── Trade history ───────────────────────────────────────────────
def trade_stats(t):
    """Win rate, average P&L, profit factor etc. for a set of closed trades."""
    wins = t.loc[t["PnL %"] > 0, "PnL %"]
    losses = t.loc[t["PnL %"] <= 0, "PnL %"]
    return pd.Series({
        "Trades": len(t),
        "Win %": round(len(wins) / len(t) * 100, 1),
        "Avg PnL %": round(t["PnL %"].mean(), 2),
        "Avg Win %": round(wins.mean(), 2) if len(wins) else 0.0,
        "Avg Loss %": round(losses.mean(), 2) if len(losses) else 0.0,
        "Profit Factor": round(wins.sum() / -losses.sum(), 2) if losses.sum() < 0 else float("inf"),
        "Avg Bars": round(t["Bars Held"].mean(), 1),
    })


st.subheader(f"Trade History ({len(trades_df)})", divider="blue")
if trades_df.empty:
    st.info("No closed trades in the downloaded history.")
else:
    first = trades_df["Entry Date"].min()
    st.caption(f"Every trade the rules closed between {first} and {diag['last_bar']} "
               f"({'6 months daily' if interval == '1d' else '2 years weekly'}). "
               "P&L is per trade, before costs; shorts are F&O stocks only.")
    hist = trades_df.drop(columns="_latest")

    overall = trade_stats(hist)
    m = st.columns(4)
    m[0].metric("Closed Trades", int(overall["Trades"]))
    m[1].metric("Win Rate", f"{overall['Win %']}%")
    m[2].metric("Avg P&L / Trade", f"{overall['Avg PnL %']}%")
    m[3].metric("Profit Factor", overall["Profit Factor"])
    m = st.columns(4)
    m[0].metric("Avg Win", f"{overall['Avg Win %']}%")
    m[1].metric("Avg Loss", f"{overall['Avg Loss %']}%")
    m[2].metric("Avg Bars Held", overall["Avg Bars"])

    by = st.radio("Break down by", ["Side", "Reason", "VPA Score", "Exit Month"],
                  horizontal=True)
    grp = hist.assign(**{"Exit Month": hist["Exit Date"].str[:7]})
    keys = ["Side", by] if by != "Side" else ["Side"]
    st.dataframe(grp.groupby(keys).apply(trade_stats, include_groups=False).reset_index(),
                 width="stretch", hide_index=True)

    f1, f2 = st.columns(2)
    side_f = f1.multiselect("Side", ["Long", "Short"], default=[])
    reason_f = f2.multiselect("Exit reason", sorted(hist["Reason"].unique()), default=[])
    view = hist
    if side_f:
        view = view[view["Side"].isin(side_f)]
    if reason_f:
        view = view[view["Reason"].isin(reason_f)]
    st.dataframe(view.sort_values("Exit Date", ascending=False).reset_index(drop=True),
                 width="stretch", hide_index=True)
