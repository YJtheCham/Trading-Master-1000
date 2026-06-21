"""
监控引擎: 定时检查告警规则并触发通知 (并行查询)
"""
import concurrent.futures
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from functools import partial
from typing import Optional

from .models import AlertRule, AlertEvent, CONDITION_TYPES
from .conditions import evaluate
from .notifier import notify, log_to_file
from .health import record_check, get_health, summary

logger = logging.getLogger(__name__)

RULES_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "alert_rules.json"
_rules_lock = threading.Lock()


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
    with _rules_lock:
        RULES_FILE.write_text(
            json.dumps([r.__dict__ for r in rules], ensure_ascii=False, indent=2))


def add_rule(rule: AlertRule) -> AlertRule:
    with _rules_lock:
        rules = load_rules()
        rules.append(rule)
        save_rules(rules)
    return rule


def remove_rule(uid: str) -> bool:
    with _rules_lock:
        rules = load_rules()
        new_rules = [r for r in rules if r.uid != uid]
        save_rules(new_rules)
    return len(new_rules) < len(rules)


def toggle_rule(uid: str, enabled: Optional[bool] = None) -> Optional[AlertRule]:
    with _rules_lock:
        rules = load_rules()
        for r in rules:
            if r.uid == uid:
                r.enabled = enabled if enabled is not None else not r.enabled
                save_rules(rules)
                return r
    return None


def _mark_triggered(rule: AlertRule, triggered_at: str):
    """线程安全: 记录规则触发时间"""
    with _rules_lock:
        rules = load_rules()
        for r in rules:
            if r.uid == rule.uid:
                r.last_triggered = triggered_at
                break
        save_rules(rules)


def _check_single_rule(rule: AlertRule) -> None:
    """在独立线程中检查单条规则"""
    from src.data.fetcher import fetch_realtime_data
    t0 = time.monotonic()
    try:
        df = fetch_realtime_data(rule.symbol, rule.market, period_days=120)
        elapsed_ms = (time.monotonic() - t0) * 1000
        if df.empty or len(df) < 5:
            record_check("monitor", rule.symbol, rule.market,
                         success=False, error="数据为空或不足5条",
                         latency_ms=elapsed_ms)
            return
        record_check("monitor", rule.symbol, rule.market,
                     success=True, latency_ms=elapsed_ms)
    except Exception as e:
        elapsed_ms = (time.monotonic() - t0) * 1000
        record_check("monitor", rule.symbol, rule.market,
                     success=False, error=str(e)[:100],
                     latency_ms=elapsed_ms)
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
        _mark_triggered(rule, event.triggered_at)


def _check_connection_alert():
    health = get_health()
    if health.should_alert():
        from .notifier import notify_connection_alert
        s = summary()
        notify_connection_alert(s)
        health.mark_alerted()


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

        to_check = []
        for rule in rules:
            if not rule.enabled:
                continue
            if rule.last_triggered:
                last = datetime.fromisoformat(rule.last_triggered)
                if now - last < timedelta(minutes=rule.cooldown_minutes):
                    continue
            to_check.append(rule)

        if not to_check:
            return

        # Wind MCP 单次 ~6s, 3 workers 并行 → 6条规则 ~12s
        max_workers = min(3, len(to_check))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            pool.map(_check_single_rule, to_check)

        _check_connection_alert()


# 全局单例
_engine: Optional[AlertEngine] = None


def get_engine() -> AlertEngine:
    global _engine
    if _engine is None:
        _engine = AlertEngine()
    return _engine
