from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


CONDITION_TYPES = {
    # 单一条件
    "above_ma":       "价格上穿均线",
    "below_ma":       "价格下穿均线",
    "above_price":    "价格超过阈值",
    "below_price":    "价格低于阈值",
    "rsi_oversold":   "RSI超卖 (<30)",
    "rsi_overbought": "RSI超买 (>70)",
    "volume_spike":   "成交量突增",
    "daily_change":   "日涨跌幅",
    "golden_cross":   "金叉 (短期MA上穿长期MA)",
    "death_cross":    "死叉 (短期MA下穿长期MA)",
    "bollinger_upper":"触及布林上轨",
    "bollinger_lower":"触及布林下轨",
    # 组合策略
    "ma_cross_combo":   "均线组合 (金叉买入/死叉卖出)",
    "rsi_combo":        "RSI组合 (超卖买入/超买卖出)",
    "bollinger_combo":  "布林组合 (下轨买入/上轨卖出)",
    "ma_rsi_combo":     "MA+RSI组合 (趋势+超买超卖联动)",
    "volume_breakout":  "放量突破 (放量+创20日新高)",
    "ma_triple":        "三均线组合 (多头排列/空头排列)",
    # GTJA 191 Alpha 因子
    "alpha120":        "Alpha120 (收盘价-VWAP偏离度)",
    "alpha006":        "Alpha006 (开量与10日量相关性反值)",
    "alpha053":        "Alpha053 (10日收盘价比率)",
    "alpha009":        "Alpha009 (5日均价下移)",
    "alpha015":        "Alpha015 (量价秩相关反值)",
}


@dataclass
class AlertRule:
    symbol: str
    market: str
    condition: str
    params: dict = field(default_factory=dict)
    label: str = ""
    enabled: bool = True
    interval_minutes: int = 5
    cooldown_minutes: int = 60
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="minutes"))
    last_triggered: Optional[str] = None
    last_pushed: Optional[str] = None
    push_count_today: int = 0
    push_date: Optional[str] = None

    @property
    def uid(self) -> str:
        return f"{self.market}_{self.symbol}_{self.condition}_{self.created_at}"

    @property
    def summary(self) -> str:
        desc = CONDITION_TYPES.get(self.condition, self.condition)
        extra = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"[{self.market}] {self.symbol} {desc} ({extra})"


@dataclass
class AlertEvent:
    rule: AlertRule
    triggered_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    current_price: float = 0.0
    message: str = ""
    action: str = ""
