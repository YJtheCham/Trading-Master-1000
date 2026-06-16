"""
监控设置: 交易时段 / 频次 / 交易日
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Optional

from src.utils.config import DATA_DIR

SETTINGS_FILE = DATA_DIR / "monitor_settings.json"


# A 股交易时段 (北京时间)
A_SESSIONS = [
    (time(9, 30), time(11, 30)),
    (time(13, 0), time(15, 0)),
]

# 港股交易时段 (北京时间)
HK_SESSIONS = [
    (time(9, 30), time(12, 0)),
    (time(13, 0), time(16, 0)),
]

# 美股交易时段 (北京时间, 冬令时)
US_SESSIONS = [
    (time(21, 30), time(4, 0)),  # 晚上9:30 - 凌晨4:00
]


SESSION_MAP = {
    "A":  A_SESSIONS,
    "HK": HK_SESSIONS,
    "US": US_SESSIONS,
}

MARKET_NAMES = {"A": "A股", "HK": "港股", "US": "美股"}


@dataclass
class MonitorSettings:
    market: str = "A"               # 参考市场时段
    interval_minutes: int = 10      # 检查间隔(分钟)
    custom_start: Optional[str] = None  # 自定义开始时间 HH:MM
    custom_end: Optional[str] = None    # 自定义结束时间 HH:MM
    trade_days_only: bool = True    # 仅交易日(周一到周五)
    enabled: bool = True

    @property
    def sessions(self) -> list[tuple[time, time]]:
        if self.custom_start and self.custom_end:
            try:
                start = time.fromisoformat(self.custom_start)
                end = time.fromisoformat(self.custom_end)
                return [(start, end)]
            except ValueError:
                pass
        return SESSION_MAP.get(self.market, A_SESSIONS)

    def is_trade_time(self, now: Optional[datetime] = None) -> bool:
        """判断当前是否在交易时段内"""
        if not self.enabled:
            return False
        now = now or datetime.now()

        # 非交易日跳过
        if self.trade_days_only and now.weekday() >= 5:
            return False

        t = now.time()
        for start, end in self.sessions:
            if start <= end:
                # 普通时段 (如 9:30-15:00)
                if start <= t <= end:
                    return True
            else:
                # 跨天时段 (如美股 21:30-04:00)
                if t >= start or t <= end:
                    return True
        return False

    @property
    def summary(self) -> str:
        market_name = MARKET_NAMES.get(self.market, self.market)
        sessions_str = "; ".join(
            f"{s.strftime('%H:%M')}-{e.strftime('%H:%M')}"
            for s, e in self.sessions
        )
        days = "交易日" if self.trade_days_only else "每天"
        return (f"{market_name} {sessions_str} | "
                f"每{self.interval_minutes}分钟 | {days}")

    def to_dict(self) -> dict:
        return {
            "market": self.market,
            "interval_minutes": self.interval_minutes,
            "custom_start": self.custom_start,
            "custom_end": self.custom_end,
            "trade_days_only": self.trade_days_only,
            "enabled": self.enabled,
        }


# ─── 持久化 ───────────────────────────────────────────────
def load_settings() -> MonitorSettings:
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text())
            return MonitorSettings(**data)
        except Exception:
            pass
    return MonitorSettings()


def save_settings(s: MonitorSettings):
    SETTINGS_FILE.write_text(
        json.dumps(s.to_dict(), ensure_ascii=False, indent=2))
