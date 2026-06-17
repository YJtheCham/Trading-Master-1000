"""
条件评估器: 检查股票数据是否满足告警规则
"""
import numpy as np
import pandas as pd

from .models import AlertRule


def evaluate(rule: AlertRule, df: pd.DataFrame) -> tuple[bool, str, str]:
    """检查规则是否触发, 返回 (触发?, 消息, 操作建议)"""
    prices = df["Close"].values
    volumes = df["Volume"].values if "Volume" in df.columns else None
    high = df["High"].values if "High" in df.columns else prices
    low = df["Low"].values if "Low" in df.columns else prices

    check = _checkers.get(rule.condition)
    if check is None:
        return False, f"未知条件: {rule.condition}", ""

    return check(prices, volumes, high, low, rule.params)


def _sma(data: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(data).rolling(window).mean().values


def _ema(data: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(data).ewm(span=window, adjust=False).mean().values


def _rsi(data: np.ndarray, window: int = 14) -> np.ndarray:
    delta = np.diff(data)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window).mean().values
    avg_loss = pd.Series(loss).rolling(window).mean().values
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([[np.nan], rsi])


def _bollinger(data: np.ndarray, window: int = 20, std: int = 2):
    ma = _sma(data, window)
    sd = pd.Series(data).rolling(window).std().values
    return ma, ma + std * sd, ma - std * sd


# ─── 各条件检查函数 ───────────────────────────────────────
def check_above_ma(prices, volumes, high, low, params):
    window = params.get("window", 20)
    ma_type = params.get("ma_type", "sma")
    ma = _sma(prices, window) if ma_type == "sma" else _ema(prices, window)
    if len(prices) < 2 or np.isnan(ma[-1]):
        return False, "", ""
    if prices[-1] > ma[-1] and prices[-2] <= ma[-2]:
        return (True,
                f"{prices[-1]:.2f} 上穿{window}日均线({ma[-1]:.2f})",
                "关注买入机会")
    return False, "", ""


def check_below_ma(prices, volumes, high, low, params):
    window = params.get("window", 20)
    ma = _sma(prices, window)
    if len(prices) < 2 or np.isnan(ma[-1]):
        return False, "", ""
    if prices[-1] < ma[-1] and prices[-2] >= ma[-2]:
        return (True,
                f"{prices[-1]:.2f} 下穿{window}日均线({ma[-1]:.2f})",
                "考虑止损或减仓")
    return False, "", ""


def check_above_price(prices, volumes, high, low, params):
    threshold = params.get("threshold", 0)
    if threshold <= 0:
        return False, "", ""
    if prices[-1] >= threshold:
        return (True,
                f"价格 {prices[-1]:.2f} 触及目标价 {threshold}",
                "达到目标价, 考虑卖出")
    return False, "", ""


def check_below_price(prices, volumes, high, low, params):
    threshold = params.get("threshold", 0)
    if threshold <= 0:
        return False, "", ""
    if prices[-1] <= threshold:
        return (True,
                f"价格 {prices[-1]:.2f} 跌破止损价 {threshold}",
                "触发止损, 建议卖出")
    return False, "", ""


def check_rsi_oversold(prices, volumes, high, low, params):
    window = params.get("window", 14)
    level = params.get("level", 30)
    rsi = _rsi(prices, window)
    if np.isnan(rsi[-1]):
        return False, "", ""
    if rsi[-1] < level:
        return (True,
                f"RSI={rsi[-1]:.1f} 进入超卖区 (<{level})",
                "超卖反弹机会, 关注买入")
    return False, "", ""


def check_rsi_overbought(prices, volumes, high, low, params):
    window = params.get("window", 14)
    level = params.get("level", 70)
    rsi = _rsi(prices, window)
    if np.isnan(rsi[-1]):
        return False, "", ""
    if rsi[-1] > level:
        return (True,
                f"RSI={rsi[-1]:.1f} 进入超买区 (>{level})",
                "超买风险, 考虑减仓")
    return False, "", ""


def check_volume_spike(prices, volumes, high, low, params):
    if volumes is None or len(volumes) < 22:
        return False, "", ""
    ratio = params.get("ratio", 2.0)
    avg_vol = np.mean(volumes[-21:-1])
    if avg_vol == 0:
        return False, "", ""
    if volumes[-1] > avg_vol * ratio:
        return (True,
                f"成交量 {volumes[-1]:.0f} 是20日均量的{volumes[-1]/avg_vol:.1f}倍",
                "放量异动, 关注突破方向")
    return False, "", ""


def check_daily_change(prices, volumes, high, low, params):
    if len(prices) < 2:
        return False, "", ""
    direction = params.get("direction", "up")
    pct = params.get("pct", 5.0)
    change = (prices[-1] - prices[-2]) / prices[-2] * 100
    if direction == "up" and change >= pct:
        return (True,
                f"日内涨幅 {change:.1f}% 超过 {pct}%",
                "强势上涨, 持仓观察")
    if direction == "down" and change <= -pct:
        return (True,
                f"日内跌幅 {change:.1f}% 超过 {pct}%",
                "大幅下跌, 关注风险")
    return False, "", ""


def check_golden_cross(prices, volumes, high, low, params):
    short_w = params.get("short", 20)
    long_w = params.get("long", 60)
    short_ma = _sma(prices, short_w)
    long_ma = _sma(prices, long_w)
    if len(prices) < 2 or np.isnan(short_ma[-1]) or np.isnan(long_ma[-1]):
        return False, "", ""
    if (short_ma[-1] > long_ma[-1] and short_ma[-2] <= long_ma[-2]):
        return (True,
                f"短期均线({short_w})上穿长期均线({long_w}) 形成金叉",
                "趋势转多, 建议买入")
    return False, "", ""


def check_death_cross(prices, volumes, high, low, params):
    short_w = params.get("short", 20)
    long_w = params.get("long", 60)
    short_ma = _sma(prices, short_w)
    long_ma = _sma(prices, long_w)
    if len(prices) < 2 or np.isnan(short_ma[-1]) or np.isnan(long_ma[-1]):
        return False, "", ""
    if (short_ma[-1] < long_ma[-1] and short_ma[-2] >= long_ma[-2]):
        return (True,
                f"短期均线({short_w})下穿长期均线({long_w}) 形成死叉",
                "趋势转空, 建议卖出")
    return False, "", ""


def check_bollinger_upper(prices, volumes, high, low, params):
    window = params.get("window", 20)
    std = params.get("std", 2)
    _, upper, _ = _bollinger(prices, window, std)
    if np.isnan(upper[-1]):
        return False, "", ""
    if prices[-1] >= upper[-1]:
        return (True,
                f"价格触及布林上轨({upper[-1]:.2f})",
                "超买区域, 注意回调风险")
    return False, "", ""


def check_bollinger_lower(prices, volumes, high, low, params):
    window = params.get("window", 20)
    std = params.get("std", 2)
    _, _, lower = _bollinger(prices, window, std)
    if np.isnan(lower[-1]):
        return False, "", ""
    if prices[-1] <= lower[-1]:
        return (True,
                f"价格触及布林下轨({lower[-1]:.2f})",
                "超卖区域, 关注反弹机会")
    return False, "", ""


def check_ma_cross_combo(prices, volumes, high, low, params):
    short_w = int(params.get("short", 20))
    long_w = int(params.get("long", 60))
    short_ma = _sma(prices, short_w)
    long_ma = _sma(prices, long_w)
    if len(prices) < 2 or np.isnan(short_ma[-1]) or np.isnan(long_ma[-1]):
        return False, "", ""
    if short_ma[-1] > long_ma[-1] and short_ma[-2] <= long_ma[-2]:
        return (True, f"金叉: {short_w}日MA({short_ma[-1]:.1f})上穿{long_w}日MA({long_ma[-1]:.1f})",
                "均线金叉, 趋势转多, 建议买入")
    if short_ma[-1] < long_ma[-1] and short_ma[-2] >= long_ma[-2]:
        return (True, f"死叉: {short_w}日MA({short_ma[-1]:.1f})下穿{long_w}日MA({long_ma[-1]:.1f})",
                "均线死叉, 趋势转空, 建议卖出")
    return False, "", ""


def check_rsi_combo(prices, volumes, high, low, params):
    window = int(params.get("window", 14))
    oversold = int(params.get("oversold", 30))
    overbought = int(params.get("overbought", 70))
    rsi = _rsi(prices, window)
    if np.isnan(rsi[-1]):
        return False, "", ""
    if rsi[-1] < oversold:
        return (True, f"RSI({rsi[-1]:.1f})进入超卖区(<{oversold}), 超卖反弹机会",
                "超卖区域, 关注买入机会")
    if rsi[-1] > overbought:
        return (True, f"RSI({rsi[-1]:.1f})进入超买区(>{overbought}), 超买回调风险",
                "超买区域, 考虑减仓")
    return False, "", ""


def check_bollinger_combo(prices, volumes, high, low, params):
    window = int(params.get("window", 20))
    std = int(params.get("std", 2))
    ma, upper, lower = _bollinger(prices, window, std)
    if np.isnan(upper[-1]):
        return False, "", ""
    if prices[-1] <= lower[-1]:
        return (True, f"价格({prices[-1]:.2f})触及布林下轨({lower[-1]:.2f})",
                "超卖区域, 关注反弹买入机会")
    if prices[-1] >= upper[-1]:
        return (True, f"价格({prices[-1]:.2f})触及布林上轨({upper[-1]:.2f})",
                "超买区域, 注意回调减仓")
    return False, "", ""


def check_ma_rsi_combo(prices, volumes, high, low, params):
    """MA趋势确认 + RSI极端值入场"""
    ma_window = int(params.get("ma_window", 60))
    rsi_window = int(params.get("rsi_window", 14))
    oversold = int(params.get("oversold", 30))
    overbought = int(params.get("overbought", 70))
    ma = _sma(prices, ma_window)
    rsi = _rsi(prices, rsi_window)
    if np.isnan(ma[-1]) or np.isnan(rsi[-1]) or len(prices) < 2:
        return False, "", ""
    price = prices[-1]
    above_ma = price > ma[-1]
    below_ma = price < ma[-1]
    prev_above = prices[-2] > ma[-2]
    prev_below = prices[-2] < ma[-2]
    # 多头趋势 + RSI超卖 → 回调买入
    if above_ma and rsi[-1] < oversold:
        return (True, f"多头趋势(价{price:.1f}>MA{ma_window})中RSI超卖({rsi[-1]:.1f}), 回调买入信号",
                "多头趋势中的超卖, 建议逢低买入")
    # 空头趋势 + RSI超买 → 反弹卖出
    if below_ma and rsi[-1] > overbought:
        return (True, f"空头趋势(价{price:.1f}<MA{ma_window})中RSI超买({rsi[-1]:.1f}), 反弹卖出信号",
                "空头趋势中的超买, 建议逢高减仓")
    # 价格上穿长期均线
    if above_ma and not prev_above:
        return (True, f"价格({price:.1f})上穿{ma_window}日均线({ma[-1]:.1f})",
                "趋势转多, 建议买入")
    # 价格下穿长期均线
    if below_ma and not prev_below:
        return (True, f"价格({price:.1f})下穿{ma_window}日均线({ma[-1]:.1f})",
                "趋势转空, 建议卖出")
    return False, "", ""


def check_volume_breakout(prices, volumes, high, low, params):
    """放量突破: 价格创20日新高 + 成交量放大2倍"""
    lookback = int(params.get("lookback", 20))
    vol_ratio = float(params.get("vol_ratio", 2.0))
    if volumes is None or len(prices) < lookback + 5:
        return False, "", ""
    recent_high = np.max(prices[-lookback:-1])
    avg_vol = np.mean(volumes[-lookback:-1])
    if avg_vol == 0:
        return False, "", ""
    if prices[-1] > recent_high and volumes[-1] > avg_vol * vol_ratio:
        return (True, f"放量突破: 价格{prices[-1]:.2f}创{lookback}日新高, 成交量{volumes[-1]/avg_vol:.1f}倍",
                "放量突破, 强势信号, 建议买入")
    return False, "", ""


def check_ma_triple(prices, volumes, high, low, params):
    """三均线多头/空头排列"""
    short_w = int(params.get("short", 10))
    mid_w = int(params.get("mid", 30))
    long_w = int(params.get("long", 60))
    s = _sma(prices, short_w)
    m = _sma(prices, mid_w)
    l = _sma(prices, long_w)
    if np.isnan(s[-1]) or np.isnan(m[-1]) or np.isnan(l[-1]):
        return False, "", ""
    # 多头排列: 短 > 中 > 长
    if s[-1] > m[-1] > l[-1]:
        return (True, f"三均线多头排列: MA{short_w}({s[-1]:.1f}) > MA{mid_w}({m[-1]:.1f}) > MA{long_w}({l[-1]:.1f})",
                "多头排列, 趋势向上, 持仓或买入")
    # 空头排列: 短 < 中 < 长
    if s[-1] < m[-1] < l[-1]:
        return (True, f"三均线空头排列: MA{short_w}({s[-1]:.1f}) < MA{mid_w}({m[-1]:.1f}) < MA{long_w}({l[-1]:.1f})",
                "空头排列, 趋势向下, 减仓或观望")
    return False, "", ""


# ─── GTJA 191 Alpha 因子 ─────────────────────────────────
def check_alpha120(prices, volumes, high, low, params):
    """Alpha120: (close-VWAP)/(close+VWAP) 排名比率"""
    if volumes is None or len(prices) < 20:
        return False, "", ""
    vwap = np.cumsum(prices[-20:] * volumes[-20:]) / np.cumsum(volumes[-20:])
    score = (prices[-1] - vwap[-1]) / (prices[-1] + vwap[-1] + 1e-8)
    threshold = params.get("threshold", 0.02)
    if score > threshold:
        return (True, f"Alpha120={score:.4f} (> {threshold}), 收盘价显著高于VWAP", "日内强势, 关注突破")
    if score < -threshold:
        return (True, f"Alpha120={score:.4f} (< {-threshold}), 收盘价显著低于VWAP", "日内弱势, 可能回调")
    return False, "", ""


def check_alpha006(prices, volumes, high, low, params):
    """Alpha006: -correlation(open_price_volume_ratio, volume, 10)"""
    if volumes is None or len(prices) < 15:
        return False, "", ""
    op = high[-10:] if hasattr(high, '__len__') else prices[-10:]
    vol = volumes[-10:]
    ratio = op / (np.maximum(vol, 1))
    corr = np.corrcoef(ratio, vol)[0, 1] if len(ratio) > 1 else 0
    score = -corr if not np.isnan(corr) else 0
    threshold = params.get("threshold", 0.3)
    if score > threshold:
        return (True, f"Alpha006={score:.3f} (> {threshold}), 量价负相关", "关注反转信号")
    return False, "", ""


def check_alpha053(prices, volumes, high, low, params):
    """Alpha053: close / close[10]"""
    if len(prices) < 15:
        return False, "", ""
    ratio = prices[-1] / (prices[-11] + 1e-8)
    threshold_up = params.get("threshold_up", 1.05)
    threshold_dn = params.get("threshold_dn", 0.95)
    if ratio > threshold_up:
        return (True, f"Alpha053={ratio:.3f} (> {threshold_up}), 10日涨幅显著", "趋势走强")
    if ratio < threshold_dn:
        return (True, f"Alpha053={ratio:.3f} (< {threshold_dn}), 10日跌幅显著", "趋势走弱")
    return False, "", ""


def check_alpha009(prices, volumes, high, low, params):
    """Alpha009: SMA(delta(close,1), 5) < 0"""
    if len(prices) < 8:
        return False, "", ""
    delta = np.diff(prices[-8:])
    sma5 = np.mean(delta[-5:])
    if sma5 < 0:
        return (True, f"Alpha009={sma5:.3f} (< 0), 5日均价动能下行", "短期弱势, 注意风险")
    if sma5 > 0.05:
        return (True, f"Alpha009={sma5:.3f} (> 0.05), 5日均价动能上行", "短期强势, 关注突破")
    return False, "", ""


_checkers = {
    "above_ma": check_above_ma, "below_ma": check_below_ma,
    "above_price": check_above_price, "below_price": check_below_price,
    "rsi_oversold": check_rsi_oversold, "rsi_overbought": check_rsi_overbought,
    "volume_spike": check_volume_spike, "daily_change": check_daily_change,
    "golden_cross": check_golden_cross, "death_cross": check_death_cross,
    "bollinger_upper": check_bollinger_upper, "bollinger_lower": check_bollinger_lower,
    "ma_cross_combo": check_ma_cross_combo, "rsi_combo": check_rsi_combo,
    "bollinger_combo": check_bollinger_combo, "ma_rsi_combo": check_ma_rsi_combo,
    "volume_breakout": check_volume_breakout, "ma_triple": check_ma_triple,
    "alpha120": check_alpha120, "alpha006": check_alpha006,
    "alpha053": check_alpha053, "alpha009": check_alpha009,
}

# ─── 通知预览 ─────────────────────────────────────────────
NOTIFICATION_PREVIEWS = {
    "above_ma":       ("价格上穿{window}日均线", "关注买入机会"),
    "below_ma":       ("价格下穿{window}日均线", "考虑止损或减仓"),
    "above_price":    ("价格触及目标价 {threshold}", "达到目标价, 考虑卖出"),
    "below_price":    ("价格跌破止损价 {threshold}", "触发止损, 建议卖出"),
    "rsi_oversold":   ("RSI进入超卖区 (<{level})", "超卖反弹机会, 关注买入"),
    "rsi_overbought": ("RSI进入超买区 (>{level})", "超买风险, 考虑减仓"),
    "volume_spike":   ("成交量放大{ratio}倍", "放量异动, 关注突破方向"),
    "daily_change":   ("日内涨跌幅达{pct}%", "关注市场波动"),
    "golden_cross":   ("{short}日均线上穿{long}日均线", "趋势转多, 建议买入"),
    "death_cross":    ("{short}日均线下穿{long}日均线", "趋势转空, 建议卖出"),
    "bollinger_upper":("价格触及布林上轨", "超买区域, 注意回调风险"),
    "bollinger_lower":("价格触及布林下轨", "超卖区域, 关注反弹机会"),
    "ma_cross_combo": ("均线交叉信号", "金叉买入 / 死叉卖出"),
    "rsi_combo":      ("RSI极端信号", "超卖买入 / 超买卖出"),
    "bollinger_combo":("布林通道信号", "下轨买入 / 上轨卖出"),
    "ma_rsi_combo":   ("MA+RSI联动信号", "趋势确认后入场"),
    "volume_breakout":("放量突破信号", "放量创新高, 建议关注"),
    "ma_triple":      ("三均线排列信号", "多头持仓 / 空头减仓"),
    "alpha120":       ("Alpha120 VWAP偏离信号", "盘价偏离加权均价"),
    "alpha006":       ("Alpha006 量价负相关信号", "关注反转"),
    "alpha053":       ("Alpha053 10日比率信号", "趋势强弱判断"),
    "alpha009":       ("Alpha009 5日动能信号", "短期方向判断"),
}


def preview_notification(condition: str, params: dict,
                         price: float = 100.0) -> tuple[str, str]:
    template = NOTIFICATION_PREVIEWS.get(condition)
    if template is None:
        return "条件触发", ""
    msg_template, action_template = template
    ctx = {"price": price, **{k: v for k, v in params.items()
                              if isinstance(v, (int, float, str))}}
    try:
        msg = msg_template.format(**ctx)
    except KeyError:
        msg = msg_template
    try:
        action = action_template.format(**ctx)
    except KeyError:
        action = action_template
    return msg, action
