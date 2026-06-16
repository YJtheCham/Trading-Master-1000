"""
通知器: 桌面通知 + 日志 + 终端输出
"""
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import AlertEvent

logger = logging.getLogger(__name__)
ALERT_LOG = Path(__file__).resolve().parent.parent.parent / "data" / "alerts.log"


def notify(event: AlertEvent, method: str = "all"):
    """发送告警通知"""
    msg = _format(event)
    log_to_file(msg)

    if method in ("desktop", "all"):
        desktop_notify(event.rule.symbol, event.message)
    if method in ("terminal", "all"):
        terminal_alert(msg)


def _format(event: AlertEvent) -> str:
    return (
        f"[{event.triggered_at}] {event.rule.summary}\n"
        f"  当前价: {event.current_price:.2f}\n"
        f"  消息: {event.message}\n"
        f"  操作: {event.action}"
    )


def log_to_file(msg: str):
    try:
        with open(ALERT_LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n" + "-" * 50 + "\n")
    except Exception as e:
        logger.warning(f"写入日志失败: {e}")


def desktop_notify(title: str, message: str):
    """macOS 弹窗通知 + 终端响铃"""
    try:
        # display dialog: 可靠的文字弹窗
        subprocess.Popen(
            ["osascript", "-e",
             f'display dialog "{message}" with title "StockPredict — {title}" '
             f'buttons {{"知道了"}} default button 1 giving up after 10'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    # 终端响铃
    try:
        print("\a", end="", flush=True)
    except Exception:
        pass


def terminal_alert(msg: str):
    """终端高亮输出"""
    print(f"\n\033[93m{'='*60}\033[0m", file=sys.stderr)
    print(f"\033[91m⚠️  交易提醒\033[0m", file=sys.stderr)
    print(msg, file=sys.stderr)
    print(f"\033[93m{'='*60}\033[0m\n", file=sys.stderr)
