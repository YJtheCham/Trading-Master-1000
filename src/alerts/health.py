"""
连接健康追踪: 记录每次数据查询的成功/失败, 持久化到 JSON

用于交易监控页面展示数据源连接状态, 连续失败时触发告警通知.
"""
import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.utils.config import DATA_DIR

HEALTH_FILE = DATA_DIR / "connection_health.json"
ALERT_COOLDOWN = timedelta(minutes=30)


@dataclass
class SourceCheck:
    source_name: str
    symbol: str
    market: str
    success: bool
    error: str = ""
    latency_ms: float = 0.0
    checked_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class ConnectionHealth:
    checks: list[dict] = field(default_factory=list)
    last_alert_at: Optional[str] = None

    def _max_checks(self) -> int:
        return 200

    def add(self, check: SourceCheck):
        self.checks.append(asdict(check))
        if len(self.checks) > self._max_checks():
            self.checks = self.checks[-self._max_checks():]
        self.save()

    def recent(self, minutes: int = 60) -> list[dict]:
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            c for c in self.checks
            if datetime.fromisoformat(c["checked_at"]) >= cutoff
        ]

    def consecutive_failures(self) -> int:
        count = 0
        for c in reversed(self.checks):
            if not c["success"]:
                count += 1
            else:
                break
        return count

    def should_alert(self) -> bool:
        if self.consecutive_failures() < 2:
            return False
        if self.last_alert_at:
            last = datetime.fromisoformat(self.last_alert_at)
            if datetime.now() - last < ALERT_COOLDOWN:
                return False
        return True

    def mark_alerted(self):
        self.last_alert_at = datetime.now().isoformat(timespec="seconds")
        self.save()

    def save(self):
        try:
            HEALTH_FILE.write_text(
                json.dumps(asdict(self), ensure_ascii=False, indent=2))
        except Exception:
            pass

    @staticmethod
    def load() -> "ConnectionHealth":
        if HEALTH_FILE.exists():
            try:
                data = json.loads(HEALTH_FILE.read_text())
                return ConnectionHealth(**data)
            except Exception:
                pass
        return ConnectionHealth()


_lock = threading.Lock()


def record_check(source_name: str, symbol: str, market: str,
                 success: bool, error: str = "", latency_ms: float = 0.0):
    with _lock:
        health = ConnectionHealth.load()
        health.add(SourceCheck(
            source_name=source_name, symbol=symbol, market=market,
            success=success, error=error, latency_ms=latency_ms,
        ))


def get_health() -> ConnectionHealth:
    return ConnectionHealth.load()


def summary() -> dict:
    h = ConnectionHealth.load()
    recent = h.recent(60)
    if not recent:
        return {
            "status": "unknown",
            "recent_checks": 0,
            "success_rate": 0.0,
            "consecutive_failures": h.consecutive_failures(),
            "last_check": None,
            "last_error": None,
        }
    successes = sum(1 for c in recent if c["success"])
    last = recent[-1]
    errors = [c["error"] for c in recent if not c["success"] and c.get("error")]
    status = "ok"
    if successes / len(recent) < 0.5:
        status = "degraded"
    if h.consecutive_failures() >= 3:
        status = "down"

    return {
        "status": status,
        "recent_checks": len(recent),
        "success_rate": round(successes / len(recent) * 100, 1),
        "consecutive_failures": h.consecutive_failures(),
        "last_check": last["checked_at"],
        "last_error": errors[-1] if errors else None,
        "last_source": last.get("source_name", ""),
        "avg_latency_ms": round(
            sum(c["latency_ms"] for c in recent) / len(recent), 0),
    }
