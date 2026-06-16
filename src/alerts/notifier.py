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
    if method in ("telegram", "all"):
        telegram_notify(f"StockPredict: {event.rule.symbol}", event.message, event.action)


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


def telegram_notify(title: str, message: str, action: str = ""):
    """通过 Telegram Bot 发手机推送"""
    from src.utils.config import get_telegram_config
    cfg = get_telegram_config()
    if not cfg["token"] or not cfg["chat_id"]:
        return
    text = f"📈 *{title}*\n{message}"
    if action:
        text += f"\n\n💡 操作建议: {action}"
    try:
        import requests
        url = f"https://api.telegram.org/bot{cfg['token']}/sendMessage"
        requests.post(url, json={
            "chat_id": cfg["chat_id"],
            "text": text,
            "parse_mode": "Markdown",
        }, timeout=5)
    except Exception:
        pass
