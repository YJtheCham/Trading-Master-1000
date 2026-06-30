"""
监控引擎: 定时检查告警规则并触发通知 (并行查询)
"""
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
_rules_lock = threading.RLock()


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
        for r in rules:
            if r.symbol == rule.symbol and r.market == rule.market \
               and r.condition == rule.condition and r.params == rule.params:
                logger.info(f"规则已存在, 跳过添加: {rule.uid}")
                return r
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


PUSH_COOLDOWN_MINUTES = 30
PUSH_DAILY_LIMIT = 5


def _can_push(rule: AlertRule) -> bool:
    """检查推送限制: 同一规则30分钟内最多推送1次, 每天最多5次"""
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    # 重置每日计数
    if rule.push_date != today:
        rule.push_count_today = 0
        rule.push_date = today

    # 每日上限
    if rule.push_count_today >= PUSH_DAILY_LIMIT:
        logger.info(f"{rule.uid} 每日推送上限({PUSH_DAILY_LIMIT})已达到, 跳过推送")
        return False

    # 30分钟冷却
    if rule.last_pushed:
        last_push = datetime.fromisoformat(rule.last_pushed)
        if now - last_push < timedelta(minutes=PUSH_COOLDOWN_MINUTES):
            remaining = PUSH_COOLDOWN_MINUTES - int((now - last_push).total_seconds() / 60)
            logger.info(f"{rule.uid} 推送冷却中({remaining}分钟后可推送), 跳过推送")
            return False

    return True


def _mark_pushed(rule: AlertRule) -> None:
    """线程安全: 记录推送时间和计数"""
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    with _rules_lock:
        rules = load_rules()
        for r in rules:
            if r.uid == rule.uid:
                r.last_pushed = now.isoformat(timespec="seconds")
                if r.push_date != today:
                    r.push_count_today = 1
                    r.push_date = today
                else:
                    r.push_count_today += 1
                break
        save_rules(rules)


def _mark_triggered(rule: AlertRule, triggered_at: str):
    """线程安全: 记录规则触发时间"""
    with _rules_lock:
        rules = load_rules()
        for r in rules:
            if r.uid == rule.uid:
                r.last_triggered = triggered_at
                break
        save_rules(rules)


def _check_rules_batch(to_check: list[AlertRule]) -> None:
    """批量检查规则: 同一股票只调一次Wind MCP, 然后检查该股票的所有规则"""
    from src.data.fetcher import fetch_realtime_data
    from collections import defaultdict

    # 按 (symbol, market) 分组
    groups: dict[tuple[str, str], list[AlertRule]] = defaultdict(list)
    for rule in to_check:
        groups[(rule.symbol, rule.market)].append(rule)

    for (symbol, market), rules in groups.items():
        t0 = time.monotonic()
        try:
            df = fetch_realtime_data(symbol, market, period_days=120)
            elapsed_ms = (time.monotonic() - t0) * 1000
            if df.empty or len(df) < 5:
                record_check("monitor", symbol, market,
                             success=False, error="数据为空或不足5条",
                             latency_ms=elapsed_ms)
                continue
            record_check("monitor", symbol, market,
                         success=True, latency_ms=elapsed_ms)
        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            record_check("monitor", symbol, market,
                         success=False, error=str(e)[:100],
                         latency_ms=elapsed_ms)
            continue

        # 用同一份数据检查该股票的所有规则
        for rule in rules:
            triggered, message, action = evaluate(rule, df)
            if triggered:
                _mark_triggered(rule, datetime.now().isoformat(timespec="seconds"))
                if _can_push(rule):
                    event = AlertEvent(
                        rule=rule,
                        current_price=float(df["Close"].iloc[-1]),
                        message=message,
                        action=action,
                    )
                    notify(event)
                    _mark_pushed(rule)


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
        # Check in-memory flag first
        if self._running:
            return True
        # Fallback: check status file written by daemon process
        try:
            from pathlib import Path
            import json
            status_file = Path(__file__).resolve().parent.parent.parent / "data" / "monitor_status.json"
            if status_file.exists():
                data = json.loads(status_file.read_text())
                # Consider running if status file says so and was updated within 5 minutes
                if data.get("running"):
                    from datetime import datetime
                    ts = datetime.fromisoformat(data.get("timestamp", "2000-01-01"))
                    if (datetime.now() - ts).total_seconds() < 300:  # 5 min
                        return True
        except Exception:
            pass
        return False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log_to_file(f"[{datetime.now().isoformat()}] 监控引擎启动")
        # Write status file for cross-process communication
        self._write_status(True)

    def stop(self):
        self._running = False
        log_to_file(f"[{datetime.now().isoformat()}] 监控引擎停止")
        # Write status file for cross-process communication
        self._write_status(False)

    def _write_status(self, running: bool):
        """Write running status to file for cross-process communication"""
        try:
            from pathlib import Path
            status_file = Path(__file__).resolve().parent.parent.parent / "data" / "monitor_status.json"
            import json
            status_file.write_text(json.dumps({
                "running": running,
                "timestamp": datetime.now().isoformat(),
                "pid": None  # Will be filled by actual daemon process
            }))
        except Exception:
            pass

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

        # 按股票分组, 同股多条规则共享一次Wind MCP调用
        _check_rules_batch(to_check)

        _check_connection_alert()


# 全局单例
_engine: Optional[AlertEngine] = None


def get_engine() -> AlertEngine:
    global _engine
    if _engine is None:
        _engine = AlertEngine()
    return _engine
