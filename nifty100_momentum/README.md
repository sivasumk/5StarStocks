# Nifty 100 Momentum Ranking Dashboard

Streamlit dashboard that ranks all 100 Nifty 100 stocks by a weighted momentum
score built from RSI, SMI, OBV, ATR, ADX, relative strength vs index, and
EMA 34 / EMA 5 trend.

## Features

- Live Nifty 100 constituent list from niftyindices.com, falling back to the
  bundled `nifty100.csv` when the live list is blocked or malformed
- 7 indicators per stock with 7-day sparkline trends
- Composite momentum score (-100 to +100)
- Full dashboard + compact ranking views
- Trade signal (Strong Long … Strong Short) and SMI signal-line crossovers
- Sidebar filters: trade signal, recent SMI cross, RSI zone, EMA status,
  ATR state, ADX strength, score range
- Top 15 long / short trade-candidate tables (unfiltered)

## Scoring weights

| Component | Weight |
|-----------|--------|
| EMA 34 + EMA 5 slope | 20% |
| Relative Strength | 20% |
| RSI | 15% |
| SMI | 15% |
| ADX + DI direction | 15% |
| OBV | 10% |
| ATR expansion | 5% |

ATR and ADX use Wilder smoothing, matching TradingView. Any component that
cannot be computed (missing data) scores 0 rather than skewing the total.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Visit https://share.streamlit.io and sign in with GitHub
2. Click **New app**
3. Repository: `sivasumk/5StarStocks`
4. Branch: `master`
5. Main file path: `nifty100_momentum/app.py`
6. Click **Deploy**

The app auto-redeploys on every push to master.
