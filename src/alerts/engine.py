"""
监控引擎: 定时检查告警规则并触发通知
"""
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .models import AlertRule, AlertEvent, CONDITION_TYPES
from .conditions import evaluate
from .notifier import notify, log_to_file

logger = logging.getLogger(__name__)

RULES_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "alert_rules.json"


# ─── 规则持久化 ───────────────────────────────────────────
def load_rules() -> list[AlertRule]:
    if not RULES_FILE.exists():
        return []
    try:
        data = json.loads(RULES_FILE.read_text())
        return [AlertRule(**r) for r in data]
    except Exception as e:
        logger.warning(f"加载规则失败: {e}")
        return []


def save_rules(rules: list[AlertRule]):
    RULES_FILE.write_text(
        json.dumps([r.__dict__ for r in rules], ensure_ascii=False, indent=2))


def add_rule(rule: AlertRule) -> AlertRule:
    rules = load_rules()
    rules.append(rule)
    save_rules(rules)
    return rule


def remove_rule(uid: str) -> bool:
    rules = load_rules()
    new_rules = [r for r in rules if r.uid != uid]
    save_rules(new_rules)
    return len(new_rules) < len(rules)


def toggle_rule(uid: str, enabled: Optional[bool] = None) -> Optional[AlertRule]:
    rules = load_rules()
    for r in rules:
        if r.uid == uid:
            r.enabled = enabled if enabled is not None else not r.enabled
            save_rules(rules)
            return r
    return None


# ─── 监控引擎 ─────────────────────────────────────────────
class AlertEngine:
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log_to_file(f"[{datetime.now().isoformat()}] 监控引擎启动")

    def stop(self):
        self._running = False
        log_to_file(f"[{datetime.now().isoformat()}] 监控引擎停止")

    def _loop(self):
        while self._running:
            try:
                from .settings import load_settings
                settings = load_settings()
                interval = max(30, settings.interval_minutes * 60)
                if settings.enabled and settings.is_trade_time():
                    self._check_once()
                else:
                    logger.debug("非交易时段, 跳过检查")
            except Exception as e:
                logger.warning(f"监控检查异常: {e}")
                interval = 60
            time.sleep(interval)

    def _check_once(self):
        rules = load_rules()
        now = datetime.now()

        for rule in rules:
            if not rule.enabled:
                continue

            # 冷却检查
            if rule.last_triggered:
                last = datetime.fromisoformat(rule.last_triggered)
                if now - last < timedelta(minutes=rule.cooldown_minutes):
                    continue

            try:
                self._check_rule(rule)
            except Exception as e:
                logger.warning(f"检查规则 {rule.uid} 失败: {e}")

    def _check_rule(self, rule: AlertRule):
        from src.data.fetcher import fetch_data

        df = fetch_data(rule.symbol, rule.market, period_days=120, use_cache=True)
        if df.empty or len(df) < 5:
            return

        triggered, message, action = evaluate(rule, df)
        if triggered:
            event = AlertEvent(
                rule=rule,
                current_price=float(df["Close"].iloc[-1]),
                message=message,
                action=action,
            )
            notify(event)

            # 更新触发时间
            rules = load_rules()
            for r in rules:
                if r.uid == rule.uid:
                    r.last_triggered = event.triggered_at
                    break
            save_rules(rules)


# 全局单例
_engine: Optional[AlertEngine] = None


def get_engine() -> AlertEngine:
    global _engine
    if _engine is None:
        _engine = AlertEngine()
    return _engine
