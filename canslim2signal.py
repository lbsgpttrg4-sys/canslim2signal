import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# -----------------------------
# Sidebar: Strategy Parameters
# -----------------------------

st.sidebar.title("Strategy Parameters")

long_sma = st.sidebar.number_input("Long SMA", value=100)
short_ema = st.sidebar.number_input("Short EMA", value=20)
rsi_window = st.sidebar.number_input("RSI Window", value=14)
vol_sma = st.sidebar.number_input("Volume SMA", value=20)
atr_window = st.sidebar.number_input("ATR Window", value=14)
adx_window = st.sidebar.number_input("ADX Window", value=14)

rsi_bull = st.sidebar.slider("RSI Bull Threshold", 0.0, 100.0, value=55.0)
rsi_bear = st.sidebar.slider("RSI Bear Threshold", 0.0, 100.0, value=45.0)
atr_risk_cut = st.sidebar.slider("ATR Risk Cutoff", 0.0, 0.1, value=0.02)
adx_trend_thr = st.sidebar.slider("ADX Trend Threshold", 0.0, 100.0, value=20.0)

bullish_position = st.sidebar.selectbox("Bullish Position", [1.0, 0.5, 0.0], index=0)
neutral_position = st.sidebar.selectbox("Neutral Position", [1.0, 0.5, 0.0], index=1)
bearish_position = st.sidebar.selectbox("Bearish Position", [1.0, 0.5, 0.0], index=2)

# -----------------------------
# Indicator Functions
# -----------------------------

def rsi(series, window):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    up_ema = pd.Series(up, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    down_ema = pd.Series(down, index=series.index).ewm(alpha=1/window, adjust=False).mean()
    rs = up_ema / down_ema.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(high, low, close, window):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()

def adx_like(high, low, close, window):
    atr_val = atr(high, low, close, window)
    price_range = close.rolling(window).max() - close.rolling(window).min()
    strength = (atr_val / price_range.replace(0, np.nan)).clip(0, 1)
    return strength.rolling(window, min_periods=1).mean() * 100

# -----------------------------
# Signal Logic
# -----------------------------

def classify_regime(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    df["SMA_long"] = close.rolling(long_sma).mean()
    df["EMA_short"] = close.ewm(span=short_ema, adjust=False).mean()
    df["RSI"] = rsi(close, rsi_window)
    df["Vol_SMA"] = vol.rolling(vol_sma).mean()
    df["UpDay"] = (close > close.shift(1)).astype(int)
    df["VolConfirm"] = ((vol > 1.1 * df["Vol_SMA"]) & (df["UpDay"] == 1)).astype(int)
    df["ATR"] = atr(high, low, close, atr_window)
    df["ADX_like"] = adx_like(high, low, close, adx_window)

    latest = df.iloc[-1]
    date = df.index[-1].date()

    trend_up = latest["Close"] > latest["SMA_long"] and latest["EMA_short"] > latest["SMA_long"]
    trend_down = latest["Close"] < latest["SMA_long"] and latest["EMA_short"] < latest["SMA_long"]
    momentum_bull = latest["RSI"] >= rsi_bull
    momentum_bear = latest["RSI"] <= rsi_bear
    vol_confirm = latest["VolConfirm"] == 1
    trend_strength_ok = latest["ADX_like"] >= adx_trend_thr
    high_vol = (latest["ATR"] / latest["Close"]) >= atr_risk_cut

    bull = trend_up and momentum_bull and (vol_confirm or trend_strength_ok) and not high_vol
    bear = trend_down and (momentum_bear or high_vol)

    if bull:
        regime = "bullish"
        signal = bullish_position
    elif bear:
        regime = "bearish"
        signal = bearish_position
    else:
        regime = "neutral"
        signal = neutral_position

    return {
        "Date": date,
        "Regime": regime,
        "Signal": signal,
        "RSI": round(latest["RSI"], 2),
        "EMA": round(latest["EMA_short"], 2),
        "SMA": round(latest["SMA_long"], 2),
        "ATR": round(latest["ATR"], 2),
        "ADX": round(latest["ADX_like"], 2),
        "Vol_SMA": round(latest["Vol_SMA"], 2)
    }

# -----------------------------
# App UI
# -----------------------------

st.title("📡 Real-Time Signal Dashboard")

tickers_input = st.text_input("Enter comma-separated tickers", value="AAPL,MSFT,NVDA")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if tickers:
    st.subheader("Signal Recommendations for Next Day")
    signal_rows = []

    max_window = max(long_sma, short_ema, rsi_window, vol_sma, atr_window, adx_window)
    lookback = max_window + 5

    for ticker in tickers:
        try:
            df = yf.download(ticker, period=f"{lookback}d", interval="1d", auto_adjust=True)
            if len(df) < lookback:
                signal_rows.append({"Ticker": ticker, "Date": "—", "Regime": "insufficient data", "Signal": "—"})
                continue

            signal_info = classify_regime(df)
            signal_info["Ticker"] = ticker
            signal_rows.append(signal_info)
        except Exception as e:
            signal_rows.append({"Ticker": ticker, "Date": "—", "Regime": "error", "Signal": str(e)})

    st.dataframe(pd.DataFrame(signal_rows)[[
        "Ticker", "Date", "Signal", "Regime", "RSI", "EMA", "SMA", "ATR", "ADX", "Vol_SMA"
    ]])
