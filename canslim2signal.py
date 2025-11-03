import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import streamlit as st
import streamlit as st
import streamlit.components.v1 as components
import json
import pandas_ta as ta
import time

st.set_page_config(layout="wide")

# -----------------------------
# Config
# -----------------------------
@dataclass
class StrategyConfig:
    long_sma: int = 100
    short_ema: int = 20
    rsi_window: int = 14
    atr_window: int = 14
    adx_window: int = 14
    rsi_bull: float = 55.0
    rsi_bear: float = 45.0
    atr_risk_cut: float = 0.02
    adx_trend_thr: float = 20.0
    short_sma_period: int = 10  # Short-term SMA period
    long_sma_period: int = 30  # Long-term SMA period
    lookback: int = 100  # Lookback for momentum, RSI, etc.

# -----------------------------
# Indicators
# -----------------------------
# def rsi(series: pd.Series, window: int) -> pd.Series:
#     delta = series.diff()
#     gain = delta.clip(lower=0)
#     loss = -delta.clip(upper=0)
#     roll_up = gain.rolling(window, min_periods=window).mean()
#     roll_down = loss.rolling(window, min_periods=window).mean()
#     rs = roll_up / roll_down.replace(0, np.nan)
#     return 100 - (100 / (1 + rs))

# def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
#     prev_close = close.shift(1)
#     tr = pd.concat([
#         (high - low),
#         (high - prev_close).abs(),
#         (low - prev_close).abs()
#     ], axis=1).max(axis=1)
#     return tr.rolling(window, min_periods=1).mean()

# def adx_like(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
#     atr_val = atr(high, low, close, window)
#     price_range = (close.rolling(window).max() - close.rolling(window).min())
#     strength = (atr_val / price_range.replace(0, np.nan)).clip(0, 1)
#     return (strength.rolling(window, min_periods=1).mean() * 100)


###################
# UPDATED manual indicator calculation with pandas_ta for correct parameters
######################
def rsi(series: pd.Series, window: int) -> pd.Series:
    """Relative Strength Index using pandas_ta"""
    return ta.rsi(series, length=window)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Average True Range using pandas_ta"""
    return ta.atr(high, low, close, length=window)

def adx_like(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """
    ADX using pandas_ta.
    Returns only the ADX column (not the +DI / -DI).
    """
    adx_df = ta.adx(high, low, close, length=window)
    return adx_df[[f"DMP_{window}", f"DMN_{window}", f"ADX_{window}"]]


# -----------------------------
# Compute score
# -----------------------------
def compute_indicators_and_score(prices: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    trend_weight = 1.0
    rsi_weight = 0.8
    risk_adj_mom_weight = 1.0
    sma_crossover_signal_weight = 1.0

    # Short-term and Long-term SMAs
    sma_short = prices.rolling(cfg.short_sma_period, min_periods=cfg.short_sma_period).mean()
    sma_long = prices.rolling(cfg.long_sma_period, min_periods=cfg.long_sma_period).mean()

    # SMA Crossover signal: 1 for short > long, -1 for short < long
    sma_crossover_signal = (sma_short > sma_long).astype(int) * 2 - 1  # 1 for uptrend, -1 for downtrend

    # RSI(lookback)
    rsi_val = rsi(prices, cfg.rsi_window)

    # Momentum (lookback-period return)
    mom = prices / prices.shift(cfg.lookback) - 1

    # Volatility (lookback-period std of returns)
    ret = prices.pct_change()
    vol = ret.rolling(cfg.lookback, min_periods=cfg.lookback).std()

    # Normalize momentum by volatility (risk-adjusted)
    risk_adj_mom = mom / (vol + 1e-9)

    # Base trend signal and RSI strength
    trend = (prices > sma_short).astype(int)  # Trend signal based on short SMA
    rsi_strength = ((rsi_val < 30).astype(int)) - ((rsi_val > 70).astype(int))

    # Score: weighted sum (tune as needed)
    score = trend_weight * trend + rsi_weight * rsi_strength + risk_adj_mom_weight * risk_adj_mom + sma_crossover_signal_weight * sma_crossover_signal

    # Shift(1) to avoid lookahead; no need to shift
    # return score.shift(1)
    return score

# -----------------------------
# Core Calculation
# -----------------------------
def compute_today_row(ticker: str, cfg: StrategyConfig):
    lookback = max(cfg.long_sma, cfg.short_ema, cfg.rsi_window,
                   cfg.atr_window, cfg.adx_window) + 5
    start_date = datetime.today() - timedelta(days=lookback*2)

    df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                     interval="1d", auto_adjust=True, progress=False)
    time.sleep(1)

    if df.empty:
        return None

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, level=1, axis=1)

    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    sma_long = close.rolling(cfg.long_sma).mean().iloc[-1]
    ema_short = close.ewm(span=cfg.short_ema, adjust=False).mean().iloc[-1]
    rsi_val = rsi(close, cfg.rsi_window).iloc[-1]
    atr_val = atr(high, low, close, cfg.atr_window).iloc[-1]
    ############
    adx_val = adx_like(high, low, close, cfg.adx_window)[f"ADX_{cfg.adx_window}"].iloc[-1]
    plus_di_val = adx_like(high, low, close, cfg.adx_window)[f"DMP_{cfg.adx_window}"].iloc[-1]
    minus_di_val = adx_like(high, low, close, cfg.adx_window)[f"DMN_{cfg.adx_window}"].iloc[-1]

    mkt_dir = plus_di_val - minus_di_val

    # --- Score Calculation ---
    score = compute_indicators_and_score(close, cfg).iloc[-1]

    # --- Regime logic ---
    trend_up = (close.iloc[-1] > sma_long) and (ema_short > sma_long)
    trend_down = (close.iloc[-1] < sma_long) and (ema_short < sma_long)
    momentum_bull = rsi_val >= cfg.rsi_bull
    momentum_bear = rsi_val <= cfg.rsi_bear
    trend_strength_ok = adx_val >= cfg.adx_trend_thr
    high_vol = (atr_val / close.iloc[-1]) >= cfg.atr_risk_cut

    bull = trend_up and momentum_bull and trend_strength_ok and (not high_vol)
    bear = trend_down and (momentum_bear or high_vol)

    if bull:
        regime = "BUY"
    elif bear:
        regime = "SELL"
    else:
        regime = "HOLD"

    latest_date = df.index[-1].strftime("%Y-%m-%d")
    live_price = yf.Ticker(ticker).fast_info['last_price']
    analyst_recco = yf.Ticker(ticker).info.get("recommendationKey") or "na"

    return {
        "Date": latest_date,
        "Ticker": ticker,
        "Regime": regime,
        "Analyst":analyst_recco,
        "Score": round(score, 2),
        "SMA_Long": round(sma_long, 2),
        "EMA_Short": round(ema_short, 2),
        "RSI": round(rsi_val, 2),
        "ATR": round(atr_val, 4),
        "ADX": round(adx_val, 2),
        "Live Price": round(live_price, 2),
        "DIR":round(mkt_dir,2),
     
    }

# -----------------------------
# Streamlit App
# -----------------------------
def app():
    st.title('Stock Trading Signals with Strategy Analysis')

    # Ask the user to input tickers
    tickers_input = st.text_input(
        "Enter Stock Tickers (comma separated, e.g. AAPL, MSFT, TSLA)"
    )

    # Display the sidebar only when the user inputs tickers
    if tickers_input:
        st.sidebar.header("Configure Strategy")

        # Input for StrategyConfig parameters
        long_sma = st.sidebar.slider("Long SMA Period", 10, 200, 100)
        short_ema = st.sidebar.slider("Short EMA Period", 5, 100, 20)
        rsi_window = st.sidebar.slider("RSI Window", 5, 50, 14)
        atr_window = st.sidebar.slider("ATR Window", 5, 50, 14)
        adx_window = st.sidebar.slider("ADX Window", 5, 50, 14)
        rsi_bull = st.sidebar.slider("RSI Bull Threshold", 50.0, 70.0, 55.0)
        rsi_bear = st.sidebar.slider("RSI Bear Threshold", 30.0, 50.0, 45.0)
        atr_risk_cut = st.sidebar.slider("ATR Risk Cut Threshold", 0.005, 0.05, 0.02)
        adx_trend_thr = st.sidebar.slider("ADX Trend Strength Threshold", 10.0, 50.0, 20.0)

        # Create the config based on sidebar input
        cfg = StrategyConfig(
            long_sma=long_sma,
            short_ema=short_ema,
            rsi_window=rsi_window,
            atr_window=atr_window,
            adx_window=adx_window,
            rsi_bull=rsi_bull,
            rsi_bear=rsi_bear,
            atr_risk_cut=atr_risk_cut,
            adx_trend_thr=adx_trend_thr
        )

        # Parse the tickers
        tickers = [x.strip() for x in tickers_input.split(",")]

        # Calculate signals for the tickers
        results = []
        for t in tickers:
            row = compute_today_row(t, cfg)
            if row:
                results.append(row)

        if results:
            # Display the results with color coding
            df = pd.DataFrame(results, columns=["Date", "Ticker", "Live Price", "Regime", "Analyst", "Score", "SMA_Long", "EMA_Short", "RSI", "ATR", "ADX","DIR"])

            # Color coding based on 'Regime' column
            def colorize(val):
                if val == "BUY":
                    return 'background-color: green; color: white'
                elif val == "SELL":
                    return 'background-color: red; color: white'
                elif val == "HOLD":
                    return 'background-color: yellow; color: black'
                return ''

            styled_df = df.style.applymap(colorize, subset=['Regime'])

            st.dataframe(styled_df, use_container_width=True)
        else:
            st.write("No valid data found for the given tickers.")

        st.title("TradingView Technical Analysis Widgets for NSE Tickers")

        # # --- User Input ---
        # tickers_input1 = st.text_input(
        #     "Enter NSE tickers (comma separated). Examples: ADANIENT.NS, RELIANCE.NS, INFY.NS",
        #     value="ADANIENT.NS, BSE.NS, RELIANCE.NS, INFY.NS, TCS.NS, HDFCBANK.NS"
        # )

        # --- Interval Dropdown ---
        interval_map = {
            "1 minute": "1m",
            "5 minutes": "5m",
            "15 minutes": "15m",
            "30 minutes": "30m",
            "1 hour": "1h",
            "2 hours": "2h",
            "4 hours": "4h",
            "1 day": "1D",
            "1 week": "1W",
            "1 month": "1M",
            "1 year" : "1Y"
        }

        selected_interval_label = st.selectbox(
            "Select Default Interval",
            list(interval_map.keys()),
            index=list(interval_map.keys()).index("1 day")
        )
        selected_interval = interval_map[selected_interval_label]

        # --- Render Widgets ---
        if tickers_input:
            tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

            if not tickers:
                st.warning("No valid tickers parsed from input.")
            else:
                st.markdown("---")
                st.write(f"Rendering {len(tickers)} TradingView widgets (default interval: **{selected_interval_label}**)")

                # Display widgets 3 per row
                for i in range(0, len(tickers), 3):
                    cols = st.columns(3)
                    for j in range(3):
                        if i + j < len(tickers):
                            t = tickers[i + j]
                            base_symbol = t.split(".")[0].upper()

                            with cols[j]:
                                st.markdown(f"#### {base_symbol}")

                                config = {
                                    "colorTheme": "light",
                                    "displayMode": "single",
                                    "isTransparent": False,
                                    "locale": "en",
                                    "interval": selected_interval,
                                    "width": "100%",
                                    "height": 450,
                                    "symbol": f"NSE:{base_symbol}",
                                    "showIntervalTabs": True
                                }

                                json_config = json.dumps(config)

                                widget_html = f"""
                                <!-- TradingView Widget BEGIN -->
                                <div class="tradingview-widget-container">
                                <div class="tradingview-widget-container__widget"></div>
                                <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js" async>
                                {json_config}
                                </script>
                                </div>
                                <!-- TradingView Widget END -->
                                """
                                components.html(widget_html, height=520, scrolling=False)


if __name__ == "__main__":
    app()
