
import os
import time
import math
import json
from datetime import datetime, timezone
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import feedparser
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# Crypto Multi-Factor Scanner — research dashboard
# Data: Binance public market/derivatives APIs + optional
# CoinGecko tokenomics + Google News RSS.
# No order placement or account/API keys are required for Binance.
# ============================================================

st.set_page_config(
    page_title="Crypto Multi-Factor Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPOT_BASES = ["https://data-api.binance.vision", "https://api1.binance.com", "https://api2.binance.com"]
FAPI_BASES = ["https://fapi.binance.com", "https://fapi1.binance.com", "https://fapi2.binance.com"]
BYBIT = "https://api.bybit.com"
COINGECKO = "https://api.coingecko.com/api/v3"

WATCHLIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "SUIUSDT", "LTCUSDT", "NEARUSDT", "SNXUSDT",
    "LINKUSDT", "BNBUSDT", "AVAXUSDT", "DOTUSDT", "TRXUSDT",
    "HBARUSDT", "APTUSDT", "ATOMUSDT", "ARBUSDT", "OPUSDT",
]

WEIGHTS = {
    "structure": 15,
    "timeframes": 10,
    "liquidity": 8,
    "spot_flow": 10,
    "futures_flow": 10,
    "oi": 10,
    "funding": 6,
    "news": 8,
    "tokenomics": 5,
    "technicals": 10,
    "volume": 8,
}
TOTAL_WEIGHT = sum(WEIGHTS.values())

# ---------- HTTP / resilient multi-provider data layer ----------
# Streamlit Community Cloud can run from a region where api.binance.com returns
# HTTP 451. Binance explicitly documents data-api.binance.vision for public
# market data, so spot requests try that host first. Derivatives fall back to
# Bybit public linear-contract data when Binance Futures is inaccessible.

@st.cache_data(ttl=20, show_spinner=False)
def get_json(url, params=None):
    r = requests.get(
        url,
        params=params,
        timeout=12,
        headers={"User-Agent": "Crypto-Multi-Factor-Scanner/2.0", "Accept": "application/json"},
    )
    r.raise_for_status()
    return r.json()

def try_json(urls, params=None):
    last = None
    for url in urls:
        try:
            return get_json(url, params), url
        except Exception as e:
            last = e
    raise RuntimeError(str(last) if last else "No data provider available")

def safe_json(url, params=None, default=None):
    try:
        return get_json(url, params)
    except Exception:
        return default

def bybit_json(path, params=None, default=None):
    try:
        d = get_json(f"{BYBIT}{path}", params)
        if d.get("retCode", 0) != 0:
            return default
        return d.get("result", {})
    except Exception:
        return default

def interval_bybit(interval):
    return {"1m":"1", "3m":"3", "5m":"5", "15m":"15", "30m":"30", "1h":"60", "2h":"120", "4h":"240", "6h":"360", "12h":"720", "1d":"D"}.get(interval, interval)

# ---------- Spot market data ----------
@st.cache_data(ttl=30, show_spinner=False)
def spot_24h():
    # Primary: Binance public market-data host; fallback: Bybit spot.
    try:
        data, _ = try_json([f"{b}/api/v3/ticker/24hr" for b in SPOT_BASES])
        rows = []
        for x in data:
            s = x.get("symbol", "")
            if not s.endswith("USDT"):
                continue
            try:
                rows.append({"symbol": s, "price": float(x["lastPrice"]), "change": float(x["priceChangePercent"]),
                             "quote_volume": float(x["quoteVolume"]), "high": float(x["highPrice"]),
                             "low": float(x["lowPrice"]), "trades": int(x.get("count", 0)), "provider":"Binance"})
            except Exception:
                pass
        if rows:
            return pd.DataFrame(rows)
    except Exception:
        pass
    d = bybit_json("/v5/market/tickers", {"category":"spot"}, default={})
    rows = []
    for x in (d or {}).get("list", []):
        s = x.get("symbol", "")
        if not s.endswith("USDT"):
            continue
        try:
            rows.append({"symbol":s, "price":float(x["lastPrice"]), "change":float(x.get("price24hPcnt",0))*100,
                         "quote_volume":float(x.get("turnover24h",0)), "high":float(x.get("highPrice24h",0)),
                         "low":float(x.get("lowPrice24h",0)), "trades":0, "provider":"Bybit fallback"})
        except Exception:
            pass
    if not rows:
        raise RuntimeError("No spot market provider returned data.")
    return pd.DataFrame(rows)

@st.cache_data(ttl=300, show_spinner=False)
def spot_exchange_symbols():
    try:
        data, _ = try_json([f"{b}/api/v3/exchangeInfo" for b in SPOT_BASES])
        return {x["symbol"]:x for x in data.get("symbols",[]) if x.get("status")=="TRADING"}
    except Exception:
        d = bybit_json("/v5/market/instruments-info", {"category":"spot","limit":1000}, default={})
        return {x["symbol"]:x for x in (d or {}).get("list",[]) if x.get("status") in ("Trading","TRADING")}

def _binance_kline_df(data):
    cols=["time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df=pd.DataFrame(data,columns=cols)
    for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df["time"]=pd.to_datetime(df["time"],unit="ms",utc=True)
    return df.sort_values("time").reset_index(drop=True)

@st.cache_data(ttl=60, show_spinner=False)
def klines_spot(symbol, interval="1h", limit=240):
    try:
        data,_=try_json([f"{b}/api/v3/klines" for b in SPOT_BASES], {"symbol":symbol,"interval":interval,"limit":limit})
        if data:
            return _binance_kline_df(data)
    except Exception:
        pass
    d=bybit_json("/v5/market/kline", {"category":"spot","symbol":symbol,"interval":interval_bybit(interval),"limit":min(limit,1000)}, default={})
    rows=(d or {}).get("list",[])
    if not rows:
        raise RuntimeError(f"No spot klines available for {symbol}.")
    # Bybit: [start, open, high, low, close, volume, turnover], newest first.
    rows=list(reversed(rows))
    df=pd.DataFrame(rows,columns=["time","open","high","low","close","volume","quote_volume"])
    for c in ["open","high","low","close","volume","quote_volume"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df["time"]=pd.to_datetime(pd.to_numeric(df["time"]),unit="ms",utc=True)
    df["close_time"]=df["time"]
    df["trades"]=0
    df["taker_buy_base"]=np.nan
    df["taker_buy_quote"]=np.nan
    df["ignore"]=0
    return df

@st.cache_data(ttl=30, show_spinner=False)
def agg_trades_spot(symbol, limit=1000):
    try:
        data,_=try_json([f"{b}/api/v3/aggTrades" for b in SPOT_BASES], {"symbol":symbol,"limit":min(limit,1000)})
        if data:
            df=pd.DataFrame(data)
            df["price"]=pd.to_numeric(df["p"],errors="coerce")
            df["qty"]=pd.to_numeric(df["q"],errors="coerce")
            df["side"]=np.where(df["m"].astype(bool),"SELL","BUY")
            df["notional"]=df["price"]*df["qty"]
            return df
    except Exception:
        pass
    d=bybit_json("/v5/market/recent-trade", {"category":"spot","symbol":symbol,"limit":min(limit,1000)}, default={})
    rows=(d or {}).get("list",[])
    if not rows:
        return pd.DataFrame()
    df=pd.DataFrame(rows)
    df["price"]=pd.to_numeric(df["price"],errors="coerce")
    df["qty"]=pd.to_numeric(df["size"],errors="coerce")
    df["side"]=df["side"].astype(str).str.upper()
    df["notional"]=df["price"]*df["qty"]
    return df

# ---------- Derivatives: Binance first, Bybit linear fallback ----------
@st.cache_data(ttl=30, show_spinner=False)
def futures_ticker(symbol):
    d=safe_json(f"{FAPI_BASES[0]}/fapi/v1/ticker/24hr", {"symbol":symbol}, default={})
    if d:
        return {"provider":"Binance Futures", "price":float(d.get("lastPrice",0)), "change":float(d.get("priceChangePercent",0)),
                "volume":float(d.get("quoteVolume",0)), "oi":None, "funding":None}
    d=bybit_json("/v5/market/tickers", {"category":"linear","symbol":symbol}, default={})
    x=(d or {}).get("list",[])
    if not x: return {}
    x=x[0]
    return {"provider":"Bybit Futures fallback", "price":float(x.get("lastPrice",0)), "change":float(x.get("price24hPcnt",0))*100,
            "volume":float(x.get("turnover24h",0)), "oi":float(x.get("openInterestValue",0) or 0), "funding":float(x.get("fundingRate",0) or 0)*100}

@st.cache_data(ttl=30, show_spinner=False)
def futures_24h():
    try:
        data=get_json(f"{FAPI_BASES[0]}/fapi/v1/ticker/24hr")
        rows=[]
        for x in data:
            s=x.get("symbol","")
            if s.endswith("USDT"):
                rows.append({"symbol":s,"f_price":float(x["lastPrice"]),"f_change":float(x["priceChangePercent"]),"f_volume":float(x["quoteVolume"]),"provider":"Binance Futures"})
        if rows: return pd.DataFrame(rows)
    except Exception: pass
    d=bybit_json("/v5/market/tickers", {"category":"linear"}, default={})
    rows=[]
    for x in (d or {}).get("list",[]):
        s=x.get("symbol","")
        if s.endswith("USDT"):
            try: rows.append({"symbol":s,"f_price":float(x["lastPrice"]),"f_change":float(x.get("price24hPcnt",0))*100,"f_volume":float(x.get("turnover24h",0)),"provider":"Bybit Futures fallback"})
            except Exception: pass
    return pd.DataFrame(rows)

@st.cache_data(ttl=30, show_spinner=False)
def futures_premium(symbol):
    try:
        d=safe_json(f"{FAPI_BASES[0]}/fapi/v1/premiumIndex", {"symbol":symbol}, default={})
        if d: return {"funding":float(d.get("lastFundingRate",0))*100,"provider":"Binance Futures"}
    except Exception: pass
    d=bybit_json("/v5/market/tickers", {"category":"linear","symbol":symbol}, default={})
    x=(d or {}).get("list",[])
    if x: return {"funding":float(x[0].get("fundingRate",0) or 0)*100,"provider":"Bybit Futures fallback"}
    return {}

@st.cache_data(ttl=30, show_spinner=False)
def futures_open_interest(symbol):
    try:
        d=safe_json(f"{FAPI_BASES[0]}/fapi/v1/openInterest", {"symbol":symbol}, default={})
        if d: return {"openInterest":float(d.get("openInterest",0)),"provider":"Binance Futures"}
    except Exception: pass
    d=bybit_json("/v5/market/tickers", {"category":"linear","symbol":symbol}, default={})
    x=(d or {}).get("list",[])
    if x: return {"openInterest":float(x[0].get("openInterestValue",0) or 0),"provider":"Bybit Futures fallback"}
    return {}

@st.cache_data(ttl=90, show_spinner=False)
def futures_oi_history(symbol, period="1h", limit=24):
    try:
        d=safe_json(f"{FAPI_BASES[0]}/futures/data/openInterestHist", {"symbol":symbol,"period":period,"limit":limit}, default=[])
        if d: return d
    except Exception: pass
    interval={"5m":"5min","15m":"15min","30m":"30min","1h":"1h","2h":"2h","4h":"4h","6h":"6h","12h":"12h","1d":"1d"}.get(period,"1h")
    d=bybit_json("/v5/market/open-interest", {"category":"linear","symbol":symbol,"intervalTime":interval,"limit":min(limit,200)}, default={})
    rows=[]
    for x in reversed((d or {}).get("list",[])):
        rows.append({"sumOpenInterest":x.get("openInterest",0),"timestamp":x.get("timestamp")})
    return rows

@st.cache_data(ttl=60, show_spinner=False)
def futures_taker_flow(symbol, period="1h", limit=24):
    # Binance taker endpoint if available; otherwise use recent derivative trades
    # from Bybit and classify by the reported aggressor side.
    try:
        d=safe_json(f"{FAPI_BASES[0]}/futures/data/takerBuySellVol", {"symbol":symbol,"period":period,"limit":limit}, default=[])
        if d: return d
    except Exception: pass
    d=bybit_json("/v5/market/recent-trade", {"category":"linear","symbol":symbol,"limit":1000}, default={})
    rows=(d or {}).get("list",[])
    if not rows: return []
    buy=sum(float(x.get("size",0) or 0)*float(x.get("price",0) or 0) for x in rows if str(x.get("side","")).upper()=="BUY")
    sell=sum(float(x.get("size",0) or 0)*float(x.get("price",0) or 0) for x in rows if str(x.get("side","")).upper()=="SELL")
    return [{"takerBuyVolValue":buy,"takerSellVolValue":sell}]

@st.cache_data(ttl=90, show_spinner=False)
def futures_klines(symbol, interval="1h", limit=240):
    try:
        data,_=try_json([f"{b}/fapi/v1/klines" for b in FAPI_BASES], {"symbol":symbol,"interval":interval,"limit":limit})
        if data: return _binance_kline_df(data)
    except Exception: pass
    d=bybit_json("/v5/market/kline", {"category":"linear","symbol":symbol,"interval":interval_bybit(interval),"limit":min(limit,1000)}, default={})
    rows=list(reversed((d or {}).get("list",[])))
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows,columns=["time","open","high","low","close","volume","quote_volume"])
    for c in ["open","high","low","close","volume","quote_volume"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["time"]=pd.to_datetime(pd.to_numeric(df["time"]),unit="ms",utc=True)
    df["close_time"]=df["time"]; df["trades"]=0; df["taker_buy_base"]=np.nan; df["taker_buy_quote"]=np.nan; df["ignore"]=0
    return df

# ---------- Indicators ----------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False).mean()
    ad = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def macd(s):
    m = ema(s, 12) - ema(s, 26)
    sig = ema(m, 9)
    return m, sig, m - sig

def enrich(df):
    x = df.copy()
    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200)
    x["rsi"] = rsi(x["close"], 14)
    x["atr"] = atr(x, 14)
    x["macd"], x["macd_signal"], x["macd_hist"] = macd(x["close"])
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    x["vol_ratio"] = x["volume"] / x["vol_ma20"].replace(0, np.nan)
    return x

# ---------- Structure / liquidity ----------
def swing_levels(df, lookback=5):
    x = df.tail(lookback * 8).copy()
    highs, lows = [], []
    for i in range(lookback, len(x)-lookback):
        hi = x["high"].iloc[i]
        lo = x["low"].iloc[i]
        if hi == x["high"].iloc[i-lookback:i+lookback+1].max():
            highs.append(hi)
        if lo == x["low"].iloc[i-lookback:i+lookback+1].min():
            lows.append(lo)
    return highs[-6:], lows[-6:]

def structure_score(df):
    x = enrich(df)
    if len(x) < 80:
        return 0, "INSUFFICIENT DATA", {}
    last = x.iloc[-1]
    h1 = x["high"].tail(40)
    l1 = x["low"].tail(40)
    recent_high = h1.max()
    recent_low = l1.min()
    # Trend stack
    trend = 1 if last.close > last.ema20 > last.ema50 else -1 if last.close < last.ema20 < last.ema50 else 0
    # Momentum / break context
    hh = last.close > x["high"].iloc[-10:-1].max()
    ll = last.close < x["low"].iloc[-10:-1].min()
    score = trend * 55
    if hh: score += 35
    if ll: score -= 35
    score = float(np.clip(score, -100, 100))
    if score >= 30:
        label = "BULLISH STRUCTURE"
    elif score <= -30:
        label = "BEARISH STRUCTURE"
    else:
        label = "RANGE / MIXED"
    highs, lows = swing_levels(x)
    return score, label, {
        "recent_high": recent_high,
        "recent_low": recent_low,
        "swing_highs": highs,
        "swing_lows": lows,
    }

def timeframe_score(symbol):
    scores = {}
    for tf, lim in [("15m", 220), ("1h", 220), ("4h", 180)]:
        try:
            df = enrich(klines_spot(symbol, tf, lim))
            last = df.iloc[-1]
            s = 0
            s += 35 if last.close > last.ema20 else -35
            s += 35 if last.ema20 > last.ema50 else -35
            s += 30 if last.rsi >= 50 else -30
            scores[tf] = float(np.clip(s, -100, 100))
        except Exception:
            scores[tf] = 0
    return float(np.mean(list(scores.values()))), scores

def liquidity_metrics(df):
    x = enrich(df)
    last = float(x["close"].iloc[-1])
    highs, lows = swing_levels(x)
    nearest_h = min([h for h in highs if h > last], default=max(highs) if highs else last)
    nearest_l = max([l for l in lows if l < last], default=min(lows) if lows else last)
    dist_h = (nearest_h / last - 1) * 100 if last else 0
    dist_l = (last / nearest_l - 1) * 100 if nearest_l else 0
    # A simple liquidity score: closer obvious swing levels = more immediate liquidity.
    score = np.clip(50 + (abs(dist_l) - abs(dist_h)) * 8, -100, 100)
    return float(score), nearest_h, nearest_l

def spot_flow(symbol):
    df = agg_trades_spot(symbol)
    if df.empty:
        return 0.0, 0.0, 0.0
    buy = df.loc[df.side == "BUY", "notional"].sum()
    sell = df.loc[df.side == "SELL", "notional"].sum()
    total = buy + sell
    delta = (buy - sell) / total * 100 if total else 0
    return float(delta), float(buy), float(sell)

def futures_flow(symbol):
    data = futures_taker_flow(symbol, "1h", 24)
    if not data:
        return 0.0, 0.0, 0.0
    buy = sum(float(x.get("takerBuyVolValue", 0) or 0) for x in data)
    sell = sum(float(x.get("takerSellVolValue", 0) or 0) for x in data)
    total = buy + sell
    delta = (buy - sell) / total * 100 if total else 0
    return float(delta), float(buy), float(sell)

def oi_metrics(symbol):
    now = futures_open_interest(symbol)
    hist = futures_oi_history(symbol, "1h", 24)
    current = float(now.get("openInterest", 0) or 0) if now else 0
    change = 0.0
    if len(hist) >= 2:
        old = float(hist[0].get("sumOpenInterest", 0) or 0)
        new = float(hist[-1].get("sumOpenInterest", 0) or 0)
        if old:
            change = (new / old - 1) * 100
    provider = now.get("provider", "Unavailable") if now else "Unavailable"
    return current, float(change), provider

def funding_metrics(symbol):
    p = futures_premium(symbol)
    return float(p.get("funding", 0) or 0), p.get("provider", "Unavailable")

# ---------- News ----------
POSITIVE_WORDS = {
    "approval": 2.0, "approved": 2.0, "adoption": 1.5, "partnership": 1.5,
    "launch": 1.0, "inflow": 1.5, "upgrade": 1.5, "bullish": 1.5,
    "etf": 1.0, "integration": 1.2, "growth": 0.8, "record": 0.8,
}
NEGATIVE_WORDS = {
    "hack": -3.0, "exploit": -3.0, "lawsuit": -2.0, "ban": -2.0,
    "outflow": -1.5, "liquidation": -1.5, "delist": -2.5, "delisting": -2.5,
    "unlock": -1.0, "bearish": -1.5, "fraud": -3.0, "investigation": -2.0,
    "downgrade": -1.0, "selloff": -1.5,
}

def news_score(symbol):
    base = symbol.replace("USDT", "")
    q = quote_plus(f"{base} crypto")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        items = []
        score = 0.0
        for e in feed.entries[:12]:
            title = e.get("title", "")
            low = title.lower()
            s = 0.0
            for w, v in POSITIVE_WORDS.items():
                if w in low: s += v
            for w, v in NEGATIVE_WORDS.items():
                if w in low: s += v
            score += s
            items.append({
                "title": title,
                "link": e.get("link", ""),
                "published": e.get("published", ""),
                "impact": round(s, 2),
            })
        return float(np.clip(score * 5, -100, 100)), items
    except Exception:
        return 0.0, []

# ---------- Tokenomics ----------
@st.cache_data(ttl=3600, show_spinner=False)
def coingecko_search(asset):
    key = os.getenv("COINGECKO_API_KEY", "").strip()
    headers = {"x-cg-demo-api-key": key} if key else {}
    try:
        r = requests.get(
            f"{COINGECKO}/search",
            params={"query": asset},
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()
        coins = r.json().get("coins", [])
        return coins[0] if coins else None
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def tokenomics(symbol):
    asset = symbol.replace("USDT", "")
    match = coingecko_search(asset)
    if not match:
        return {"available": False, "reason": "CoinGecko mapping unavailable"}
    coin_id = match.get("id")
    key = os.getenv("COINGECKO_API_KEY", "").strip()
    headers = {"x-cg-demo-api-key": key} if key else {}
    try:
        r = requests.get(
            f"{COINGECKO}/coins/{coin_id}",
            params={"localization": "false", "tickers": "false", "market_data": "true",
                    "community_data": "false", "developer_data": "false"},
            headers=headers,
            timeout=12,
        )
        r.raise_for_status()
        d = r.json()
        md = d.get("market_data", {})
        return {
            "available": True,
            "name": d.get("name", asset),
            "id": coin_id,
            "market_cap": md.get("market_cap", {}).get("usd"),
            "circulating": md.get("circulating_supply"),
            "total_supply": md.get("total_supply"),
            "max_supply": md.get("max_supply"),
            "fdv": md.get("fully_diluted_valuation", {}).get("usd"),
            "ath": md.get("ath", {}).get("usd"),
            "atl": md.get("atl", {}).get("usd"),
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}

# ---------- Decision engine ----------
def clamp(v):
    return float(np.clip(v, -100, 100))

def score_direction(metrics):
    components = {
        "structure": metrics["structure_score"],
        "timeframes": metrics["timeframe_score"],
        "liquidity": metrics["liquidity_score"],
        "spot_flow": metrics["spot_delta"],
        "futures_flow": metrics["futures_delta"],
        "oi": clamp(metrics["oi_change"] * 10),
        # Positive funding is not automatically bullish: very high positive funding
        # is treated as crowded-long risk.
        "funding": clamp(-metrics["funding_rate"] * 180),
        "news": metrics["news_score"],
        "tokenomics": metrics["tokenomics_score"],
        "technicals": metrics["technical_score"],
        "volume": metrics["volume_score"],
    }
    weighted = sum(components[k] * WEIGHTS[k] for k in components) / TOTAL_WEIGHT
    weighted = clamp(weighted)

    # Require some agreement before calling BUY/SELL.
    bull_votes = sum(1 for v in components.values() if v >= 25)
    bear_votes = sum(1 for v in components.values() if v <= -25)

    if weighted >= 28 and bull_votes >= 5:
        decision = "BUY SETUP"
    elif weighted <= -28 and bear_votes >= 5:
        decision = "SELL SETUP"
    else:
        decision = "HOLD / WAIT"

    return weighted, decision, components

def technical_score(df):
    x = enrich(df)
    z = x.iloc[-1]
    s = 0
    s += 25 if z.close > z.ema20 else -25
    s += 25 if z.ema20 > z.ema50 else -25
    s += 20 if z.ema50 > z.ema200 else -20
    s += 20 if 50 <= z.rsi <= 70 else 10 if z.rsi > 70 else -15 if z.rsi < 35 else -5
    s += 10 if z.macd_hist > 0 else -10
    return float(clamp(s))

def volume_score(df):
    x = enrich(df)
    z = x.iloc[-1]
    ratio = float(z.vol_ratio) if pd.notna(z.vol_ratio) else 1
    direction = 1 if z.close > z.open else -1
    if ratio >= 2.0:
        return 70 * direction
    if ratio >= 1.3:
        return 35 * direction
    if ratio <= 0.7:
        return -10
    return 0

def tokenomics_score(t):
    if not t.get("available"):
        return 0
    circ = t.get("circulating")
    total = t.get("total_supply")
    maxs = t.get("max_supply")
    if circ and total and total > 0:
        ratio = circ / total
        if ratio >= 0.85: return 35
        if ratio >= 0.60: return 15
        if ratio < 0.25: return -40
        return -10
    if circ and maxs and maxs > 0:
        ratio = circ / maxs
        if ratio >= 0.85: return 30
        if ratio < 0.25: return -35
    return 0

def plan_levels(df, decision):
    x = enrich(df)
    p = float(x["close"].iloc[-1])
    a = float(x["atr"].iloc[-1])
    if not np.isfinite(a) or a <= 0:
        a = p * 0.01
    _, hi, lo = liquidity_metrics(x)
    if decision == "BUY SETUP":
        entry = p
        sl = min(lo * 0.995, p - 1.25*a)
        risk = max(entry - sl, 0.5*a)
        targets = [entry + 1.0*risk, entry + 1.8*risk, entry + 2.8*risk]
    elif decision == "SELL SETUP":
        entry = p
        sl = max(hi * 1.005, p + 1.25*a)
        risk = max(sl - entry, 0.5*a)
        targets = [entry - 1.0*risk, entry - 1.8*risk, entry - 2.8*risk]
    else:
        entry = p
        sl = np.nan
        targets = [np.nan, np.nan, np.nan]
    return entry, sl, targets

def analyze(symbol):
    spot = enrich(klines_spot(symbol, "1h", 240))
    structure_s, structure_label, levels = structure_score(spot)
    tf_s, tf_parts = timeframe_score(symbol)
    liq_s, liq_h, liq_l = liquidity_metrics(spot)
    sdelta, sbuy, ssell = spot_flow(symbol)
    fdelta, fbuy, fsell = futures_flow(symbol)
    oi, oichange, oi_provider = oi_metrics(symbol)
    funding, funding_provider = funding_metrics(symbol)
    nscore, news = news_score(symbol)
    tok = tokenomics(symbol)
    tscore = tokenomics_score(tok)
    tech = technical_score(spot)
    vol = volume_score(spot)

    m = {
        "structure_score": structure_s,
        "structure_label": structure_label,
        "timeframe_score": tf_s,
        "timeframes": tf_parts,
        "liquidity_score": liq_s,
        "liquidity_high": liq_h,
        "liquidity_low": liq_l,
        "spot_delta": sdelta,
        "spot_buy": sbuy,
        "spot_sell": ssell,
        "futures_delta": fdelta,
        "futures_buy": fbuy,
        "futures_sell": fsell,
        "oi": oi,
        "oi_change": oichange,
        "funding_rate": funding,
        "funding_provider": funding_provider,
        "oi_provider": oi_provider,
        "news_score": nscore,
        "news": news,
        "tokenomics": tok,
        "tokenomics_score": tscore,
        "technical_score": tech,
        "volume_score": vol,
        "price": float(spot["close"].iloc[-1]),
        "rsi": float(spot["rsi"].iloc[-1]),
        "atr": float(spot["atr"].iloc[-1]),
        "volume_ratio": float(spot["vol_ratio"].iloc[-1]) if pd.notna(spot["vol_ratio"].iloc[-1]) else 1,
    }
    total, decision, comps = score_direction(m)
    m["total_score"] = total
    m["decision"] = decision
    m["components"] = comps
    entry, sl, targets = plan_levels(spot, decision)
    m["entry"], m["sl"], m["targets"] = entry, sl, targets

    # Gainer pullback/crash-risk engine. It is a risk flag, not a prediction.
    pull = 0
    reasons = []
    change = float(spot["close"].iloc[-1] / spot["close"].iloc[-25] - 1) * 100 if len(spot) > 25 else 0
    if change > 15:
        pull += 25; reasons.append("strong short-term extension")
    if m["rsi"] > 75:
        pull += 25; reasons.append("RSI is overextended")
    if m["volume_ratio"] > 2:
        pull += 10; reasons.append("unusually high volume")
    if funding > 0.05:
        pull += 15; reasons.append("positive funding can indicate crowded longs")
    if oichange > 8 and change > 8:
        pull += 15; reasons.append("price and OI rising together: leverage is building")
    if spot < spot["ema20"].iloc[-1]:
        pull += 15; reasons.append("price lost the 20-EMA")
    m["pullback_risk"] = int(min(pull, 100))
    m["pullback_reasons"] = reasons or ["No major extension flag from the current rule set."]
    return m, spot

# ---------- BTC regime ----------
@st.cache_data(ttl=45, show_spinner=False)
def btc_regime():
    df = enrich(klines_spot("BTCUSDT", "1h", 240))
    z = df.iloc[-1]
    if z.close > z.ema50 and z.ema50 > z.ema200:
        label = "BULLISH"
    elif z.close < z.ema50 and z.ema50 < z.ema200:
        label = "BEARISH"
    else:
        label = "UNCERTAIN"
    return label, df

# ---------- BTC correlation ----------
@st.cache_data(ttl=180, show_spinner=False)
def btc_correlations():
    btc = klines_spot("BTCUSDT", "1h", 168)
    bret = btc["close"].pct_change().dropna()
    rows = []
    for s in WATCHLIST:
        if s == "BTCUSDT":
            continue
        try:
            x = klines_spot(s, "1h", 168)
            ret = x["close"].pct_change().dropna()
            n = min(len(bret), len(ret))
            corr = float(bret.tail(n).corr(ret.tail(n)))
            rows.append({"symbol": s, "corr_7d_1h": corr})
        except Exception:
            pass
    return pd.DataFrame(rows).sort_values("corr_7d_1h", ascending=False)

# ---------- UI ----------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.small-muted {color:#7f8794;font-size:.85rem;}
.metric-card {border:1px solid rgba(128,128,128,.22);border-radius:14px;padding:14px 16px;background:rgba(128,128,128,.05);}
.signal {font-size:1.45rem;font-weight:800;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Crypto Multi-Factor Scanner")
st.caption(
    "Research dashboard — live public market data. The score is a transparent weighted research score, "
    "not a win probability and not an order-execution system."
)

with st.sidebar:
    st.header("Controls")
    refresh_sec = st.slider("Auto refresh (seconds)", 30, 300, 60, 15)
    st.caption("Use the Refresh button for an immediate update.")
    if st.button("🔄 Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.subheader("Decision weights")
    st.write({k: f"{v}%" for k, v in WEIGHTS.items()})
    st.divider()
    st.caption("Optional: set COINGECKO_API_KEY as an environment variable to enable tokenomics data.")

# Automatic refresh. Data caches remain short-lived so the dashboard updates
# without hammering Binance on every UI redraw.
st_autorefresh(interval=refresh_sec * 1000, key="crypto_scanner_refresh")

try:
    regime, btc_df = btc_regime()
    tick = spot_24h()
except Exception as e:
    st.error(f"Market data could not be loaded from the primary/fallback providers: {e}")
    st.info("The app is designed to use Binance public spot data first and a public Bybit fallback when a deployment IP receives Binance HTTP 451.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("BTC Regime", regime)
with c2:
    st.metric("BTC Price", f"${btc_df.close.iloc[-1]:,.2f}")
with c3:
    st.metric("BTC 1h RSI", f"{rsi(btc_df.close).iloc[-1]:.1f}")
with c4:
    st.metric("USDT pairs scanned", f"{len(tick):,}")

st.caption("Spot provider: " + str(tick["provider"].iloc[0]) + " · Derivatives provider is shown in the detail panel.")

st.divider()

# Top gainers / losers
gainers = tick.sort_values("change", ascending=False).head(10).reset_index(drop=True)
losers = tick.sort_values("change", ascending=True).head(10).reset_index(drop=True)

left, right = st.columns(2)
with left:
    st.subheader("🚀 Top 10 Gainers")
    gain_view = gainers[["symbol","change","price","quote_volume"]].copy()
    gain_view.columns = ["Symbol","24h %","Price","24h Quote Volume"]
    ge = st.dataframe(
        gain_view,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="gainers_table",
        column_config={
            "24h %": st.column_config.NumberColumn(format="%.2f%%"),
            "Price": st.column_config.NumberColumn(format="%.8f"),
            "24h Quote Volume": st.column_config.NumberColumn(format="%.2f"),
        },
    )
with right:
    st.subheader("📉 Top 10 Losers")
    loss_view = losers[["symbol","change","price","quote_volume"]].copy()
    loss_view.columns = ["Symbol","24h %","Price","24h Quote Volume"]
    le = st.dataframe(
        loss_view,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="losers_table",
        column_config={
            "24h %": st.column_config.NumberColumn(format="%.2f%%"),
            "Price": st.column_config.NumberColumn(format="%.8f"),
            "24h Quote Volume": st.column_config.NumberColumn(format="%.2f"),
        },
    )

selected = None
try:
    if ge.selection.rows:
        selected = gainers.iloc[ge.selection.rows[0]]["symbol"]
except Exception:
    pass
try:
    if selected is None and le.selection.rows:
        selected = losers.iloc[le.selection.rows[0]]["symbol"]
except Exception:
    pass

st.divider()
st.subheader("🔗 BTC-connected market")
st.caption("The table is correlation-based: 7-day rolling 1h return correlation with BTC, not a claim that the assets always move together.")
try:
    corr_df = btc_correlations()
    st.dataframe(
        corr_df.style.format({"corr_7d_1h": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
except Exception as e:
    st.warning(f"BTC correlation section unavailable: {e}")

# Manual coin selector keeps detail usable on mobile as well.
all_symbols = sorted(set(gainers.symbol.tolist() + losers.symbol.tolist() + WATCHLIST))
default_index = all_symbols.index(selected) if selected in all_symbols else 0
manual = st.selectbox(
    "Open detailed analysis",
    all_symbols,
    index=default_index,
    format_func=lambda s: s.replace("USDT", "/USDT"),
)
if selected is not None:
    st.info(f"Selected from table: **{selected}**. The selector below is synced to the selected coin when Streamlit reruns.")
    manual = selected

st.divider()
st.subheader(f"🔍 Detailed analysis — {manual}")

with st.spinner("Fetching multi-factor data..."):
    try:
        m, chart_df = analyze(manual)
    except Exception as e:
        st.error(f"Could not analyze {manual}: {e}")
        st.stop()

d1, d2, d3, d4 = st.columns(4)
with d1:
    st.metric("Research signal", m["decision"])
with d2:
    st.metric("Weighted score", f"{m['total_score']:.1f}/100")
with d3:
    st.metric("Spot flow Δ", f"{m['spot_delta']:+.1f}%")
with d4:
    st.metric("Futures flow Δ", f"{m['futures_delta']:+.1f}%")

st.markdown(
    f"**Structure:** {m['structure_label']}  ·  **RSI:** {m['rsi']:.1f}  ·  "
    f"**OI change (24h sample):** {m['oi_change']:+.2f}%  ·  **Funding:** {m['funding_rate']:+.4f}%  ·  "
    f"**OI source:** {m['oi_provider']} · **Funding source:** {m['funding_provider']}"
)

# Price chart
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    row_heights=[0.72, 0.28])
fig.add_trace(go.Candlestick(
    x=chart_df.time, open=chart_df.open, high=chart_df.high,
    low=chart_df.low, close=chart_df.close, name="Price"
), row=1, col=1)
for col, name in [("ema20","EMA20"),("ema50","EMA50"),("ema200","EMA200")]:
    fig.add_trace(go.Scatter(x=chart_df.time, y=chart_df[col], name=name, mode="lines"), row=1, col=1)
fig.add_trace(go.Bar(x=chart_df.time, y=chart_df.volume, name="Volume"), row=2, col=1)
fig.update_layout(height=620, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Signal components
st.subheader("🧠 Factor breakdown")
factor_df = pd.DataFrame({
    "Factor": list(m["components"].keys()),
    "Score": list(m["components"].values()),
})
st.dataframe(
    factor_df.style.format({"Score":"{:.1f}"}).background_gradient(subset=["Score"], cmap="RdYlGn", vmin=-100, vmax=100),
    use_container_width=True,
    hide_index=True,
)

a, b, c = st.columns(3)
with a:
    st.subheader("Structure & liquidity")
    st.write(f"Structure score: **{m['structure_score']:+.1f}**")
    st.write(f"Nearest swing high/liquidity: `{m['liquidity_high']:.8g}`")
    st.write(f"Nearest swing low/liquidity: `{m['liquidity_low']:.8g}`")
    st.write(f"Liquidity score: **{m['liquidity_score']:+.1f}**")
with b:
    st.subheader("Flows / derivatives")
    st.write(f"Spot buy: `${m['spot_buy']:,.0f}`")
    st.write(f"Spot sell: `${m['spot_sell']:,.0f}`")
    st.write(f"Futures buy: `${m['futures_buy']:,.0f}`")
    st.write(f"Futures sell: `${m['futures_sell']:,.0f}`")
    st.write(f"Open interest: `{m['oi']:,.2f}`")
with c:
    st.subheader("Technical / volume")
    st.write(f"15m: **{m['timeframes'].get('15m',0):+.0f}**")
    st.write(f"1h: **{m['timeframes'].get('1h',0):+.0f}**")
    st.write(f"4h: **{m['timeframes'].get('4h',0):+.0f}**")
    st.write(f"Volume ratio: **{m['volume_ratio']:.2f}×**")
    st.write(f"Technical score: **{m['technical_score']:+.1f}**")

st.subheader("📐 Research levels")
if m["decision"] == "HOLD / WAIT":
    st.info("No entry plan is generated while the factors remain mixed. Wait for factor agreement.")
else:
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Reference entry", f"{m['entry']:.8g}")
    p2.metric("Invalidation / SL", f"{m['sl']:.8g}")
    p3.metric("Target 1", f"{m['targets'][0]:.8g}")
    p4.metric("Target 2", f"{m['targets'][1]:.8g}")
    p5.metric("Target 3", f"{m['targets'][2]:.8g}")
    st.caption("These are algorithmic research levels derived from current ATR/swing structure; they are not guaranteed outcomes.")

st.subheader("📰 News engine")
if m["news"]:
    news_df = pd.DataFrame(m["news"])
    st.dataframe(news_df[["title","published","impact"]], use_container_width=True, hide_index=True)
else:
    st.info("No news items were returned by the RSS source at this refresh.")

st.subheader("🧬 Tokenomics / supply")
tok = m["tokenomics"]
if tok.get("available"):
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Market cap", f"${tok['market_cap']:,.0f}" if tok.get("market_cap") else "N/A")
    tc2.metric("Circulating", f"{tok['circulating']:,.0f}" if tok.get("circulating") else "N/A")
    tc3.metric("Total supply", f"{tok['total_supply']:,.0f}" if tok.get("total_supply") else "N/A")
    tc4.metric("Max supply", f"{tok['max_supply']:,.0f}" if tok.get("max_supply") else "N/A")
else:
    st.warning(
        "Tokenomics is unavailable for this coin in the current configuration. "
        "Add a CoinGecko Demo API key as COINGECKO_API_KEY to enable it."
    )

st.subheader("⚠️ Gainer pullback / crash-risk monitor")
st.metric("Pullback-risk flag", f"{m['pullback_risk']}/100")
for r in m["pullback_reasons"]:
    st.write("•", r)
st.caption(
    "This section flags conditions associated with overextension; it does not predict that a crash will occur."
)

st.subheader("🧪 Why the signal was produced")
for k, v in m["components"].items():
    st.write(f"• **{k.replace('_',' ').title()}**: {v:+.1f}/100")

st.divider()
st.caption(
    "Data architecture: Binance public spot market-data endpoints first; Bybit public spot/linear derivatives fallback "
    "when the deployment network is restricted; optional CoinGecko token data; Google News RSS headlines. "
    "HTTP 451 is a provider/location restriction, not a malformed kline request. Always verify critical data at the source."
)
