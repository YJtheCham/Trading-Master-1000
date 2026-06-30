"""
特征工程: 从 OHLCV 数据生成多维特征矩阵

输入: DataFrame (Date, Open, High, Low, Close, Volume)
输出: DataFrame (N行 × M特征), 每行特征预测下一日 Close

特征分类:
  A. 价格衍生: 收益率、波动率、价格位置
  B. 技术指标: RSI, MACD, 布林带, ATR, OBV, MFI, ADX
  C. 成交量: 量比、OBV变化
"""
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少列: {missing}")

    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    vol = df["Volume"].values.astype(float)

    n = len(df)
    feats = {}

    # ── A1. 基础价格特征 ──
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        feats[col] = df[col].values

    # ── A2. 收益率 (多周期) ──
    for period in [1, 3, 5, 10, 20]:
        feats[f"ret_{period}d"] = _pct_change(close, period)

    # ── A3. 波动率 ──
    for period in [5, 10, 20]:
        rets = _pct_change(close, 1)
        feats[f"vol_{period}d"] = _rolling_std(rets, period)

    # ── A4. 价格位置 ──
    for period in [10, 20, 60]:
        h = _rolling_max(high, period)
        l = _rolling_min(low, period)
        rng = h - l
        feats[f"pos_{period}d"] = np.where(rng > 0, (close - l) / rng, 0.5)

    # ── A5. 量价关系 ──
    avg_vol_5 = _rolling_mean(vol, 5)
    avg_vol_20 = _rolling_mean(vol, 20)
    feats["vol_ratio_5"] = np.where(avg_vol_5 > 0, vol / avg_vol_5, 1)
    feats["vol_ratio_20"] = np.where(avg_vol_20 > 0, vol / avg_vol_20, 1)

    # ── A6. K线形态 ──
    feats["body_pct"] = np.where(close > 0, (close - df["Open"].values) / close, 0)
    feats["upper_shadow"] = np.where(close > 0, (high - np.maximum(close, df["Open"].values)) / close, 0)
    feats["lower_shadow"] = np.where(close > 0, (np.minimum(close, df["Open"].values) - low) / close, 0)

    # ── A7. 量价相关性 ──
    feats["price_vol_corr"] = _rolling_corr(close, vol, 10)

    # ── A8. 新闻情绪因子 ── (需 news_fetcher 先合并到 df)
    if "news_sent_mean" in df.columns:
        feats["news_sent_mean"] = df["news_sent_mean"].values
        feats["news_sent_std"] = df["news_sent_std"].values
        feats["news_count"] = df["news_count"].fillna(0).values
        sent_arr = feats["news_sent_mean"]
        feats["news_momentum_5d"] = _pct_change(sent_arr, 5)
    else:
        feats["news_sent_mean"] = np.zeros(n)
        feats["news_sent_std"] = np.zeros(n)
        feats["news_count"] = np.zeros(n)
        feats["news_momentum_5d"] = np.zeros(n)

    # ── B1. RSI (14) ──
    feats["rsi_14"] = _rsi(close, 14)

    # ── B2. MACD ──
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    feats["macd"] = macd_line
    feats["macd_signal"] = macd_signal
    feats["macd_hist"] = macd_line - macd_signal

    # ── B3. 布林带 %B ──
    ma20 = _rolling_mean(close, 20)
    std20 = _rolling_std(close, 20)
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    feats["bb_pct_b"] = np.where(upper - lower > 0, (close - lower) / (upper - lower), 0.5)
    feats["bb_width"] = np.where(ma20 > 0, (upper - lower) / ma20, 0)

    # ── B4. ATR (14) ──
    feats["atr_14"] = _atr(high, low, close, 14)
    feats["atr_pct"] = np.where(close > 0, feats["atr_14"] / close, 0)

    # ── B5. OBV ──
    obv = _obv(close, vol)
    feats["obv"] = obv
    feats["obv_chg_5"] = _pct_change(obv, 5)

    # ── B6. MFI (14) ──
    feats["mfi_14"] = _mfi(high, low, close, vol, 14)

    # ── B7. ADX (14) ──
    feats["adx_14"] = _adx(high, low, close, 14)

    # ── B8. 随机指标 ──
    k, d = _stochastic(high, low, close, 14, 3)
    feats["stoch_k"] = k
    feats["stoch_d"] = d

    # ── B9. 移动平均偏离 ──
    for period in [5, 10, 20, 60]:
        ma = _rolling_mean(close, period)
        feats[f"ma_dev_{period}d"] = np.where(ma > 0, (close - ma) / ma, 0)

    # ── 构建 DataFrame ──
    result = pd.DataFrame(feats, index=df.index)
    # 去掉前60行 (需要足够历史计算所有指标)
    result = result.iloc[60:].reset_index(drop=True)
    return result


# ─── 底层计算函数 ─────────────────────────────────────────
def _pct_change(arr, period):
    result = np.full(len(arr), np.nan)
    result[period:] = (arr[period:] - arr[:-period]) / np.where(arr[:-period] != 0, arr[:-period], 1)
    return result

def _rolling_mean(arr, window):
    s = pd.Series(arr)
    return s.rolling(window, min_periods=1).mean().values

def _rolling_std(arr, window):
    s = pd.Series(arr)
    return s.rolling(window, min_periods=1).std().values

def _rolling_max(arr, window):
    s = pd.Series(arr)
    return s.rolling(window, min_periods=1).max().values

def _rolling_min(arr, window):
    s = pd.Series(arr)
    return s.rolling(window, min_periods=1).min().values

def _rolling_corr(a, b, window):
    result = np.full(len(a), np.nan)
    for i in range(window, len(a)):
        result[i] = np.corrcoef(a[i-window:i], b[i-window:i])[0, 1]
        if np.isnan(result[i]):
            result[i] = 0
    return result

def _ema(arr, period):
    s = pd.Series(arr)
    return s.ewm(span=period, adjust=False).mean().values

def _rsi(close, period):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = _ema(gain, period)
    avg_loss = _ema(loss, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    return 100 - (100 / (1 + rs))

def _atr(high, low, close, period):
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    return _ema(tr, period)

def _obv(close, vol):
    obv = np.zeros(len(close))
    obv[0] = vol[0]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + vol[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - vol[i]
        else:
            obv[i] = obv[i-1]
    return obv

def _mfi(high, low, close, vol, period):
    typical = (high + low + close) / 3
    money_flow = typical * vol
    pos_flow = np.where(typical > np.roll(typical, 1), money_flow, 0)
    neg_flow = np.where(typical < np.roll(typical, 1), money_flow, 0)
    pos_flow[0] = 0
    neg_flow[0] = 0
    pos_sum = _rolling_mean(pos_flow, period) * period
    neg_sum = _rolling_mean(neg_flow, period) * period
    with np.errstate(divide="ignore", invalid="ignore"):
        mfr = np.where(neg_sum > 0, pos_sum / neg_sum, 100)
    return 100 - (100 / (1 + mfr))

def _adx(high, low, close, period):
    tr = _rolling_mean(_atr(high, low, close, 1), period) * period
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = np.where(tr > 0, 100 * _ema(plus_dm, period) / tr, 0)
    minus_di = np.where(tr > 0, 100 * _ema(minus_dm, period) / tr, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(plus_di + minus_di > 0,
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    return _ema(dx, period)

def _stochastic(high, low, close, k_period, d_period):
    low_min = _rolling_min(low, k_period)
    high_max = _rolling_max(high, k_period)
    rng = high_max - low_min
    k = np.where(rng > 0, 100 * (close - low_min) / rng, 50)
    d = _rolling_mean(k, d_period)
    return k, d
