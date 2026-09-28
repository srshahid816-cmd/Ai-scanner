# Crypto Multi-Factor Scanner

A Streamlit research dashboard for public crypto market data.

## Why the previous deployment showed HTTP 451

The previous build called `https://api.binance.com/api/v3/klines` directly. Binance can return HTTP **451** when the server/deployment IP is in a restricted location. Binance's documentation specifically provides `data-api.binance.vision` for public market data.

This build changes the architecture:

1. **Binance public spot data** → `data-api.binance.vision` first, then Binance API mirrors.
2. **Bybit public spot fallback** → used only if Binance spot data is unavailable.
3. **Binance USDⓈ-M Futures** → tried first for derivatives.
4. **Bybit public linear derivatives fallback** → supplies futures price, funding, OI and recent trade-flow data when Binance Futures is blocked.
5. **CoinGecko** → optional tokenomics/market-cap data.
6. **Google News RSS** → headline-based news context.

The app never fabricates missing data. If a provider is unavailable, the affected field is marked unavailable or the fallback provider is shown.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Upload the repository to GitHub and deploy the repository's `app.py` with Streamlit Community Cloud.

## Optional CoinGecko key

Set `COINGECKO_API_KEY` in Streamlit Secrets/environment variables to improve tokenomics access.

## Data disclaimer

This is a research dashboard, not an order-execution system. BUY/SELL/HOLD is a transparent rule-based research classification. It is not a guaranteed outcome or a win probability.

## Main factors

- Market structure
- 15m / 1h / 4h alignment
- Liquidity/swing levels
- Spot aggressive-flow proxy
- Futures aggressive-flow proxy
- Open interest and change
- Funding rate
- News headline context
- Token supply/market cap
- EMA / RSI / MACD / ATR
- Volume expansion
- Algorithmic reference entry, invalidation and targets
- Gainer pullback-risk flags
- BTC 7-day 1h correlation watchlist
