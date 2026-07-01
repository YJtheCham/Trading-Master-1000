#!/usr/bin/env python3
"""
后台监控进程: 不依赖 Streamlit, 定时检查告警规则并发送通知
用法: python3 tools/monitor_daemon.py
终止: Ctrl+C
"""
import sys, time, signal, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alerts.engine import get_engine, load_rules
from src.alerts.settings import load_settings

running = True

def stop(sig, frame):
    global running
    running = False
    print("\n⏹ 停止监控...")

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

engine = get_engine()
engine.start()
print("🟢 交易监控已启动 (后台进程)")
settings = load_settings()
rules = load_rules()
print(f"   时段: {settings.summary}")
print(f"   规则: {len(rules)} 条 (启用 {sum(1 for r in rules if r.enabled)} 条)")

# Update status file with PID
try:
    import json
    from datetime import datetime
    status_file = Path(__file__).resolve().parent.parent / "data" / "monitor_status.json"
    status_file.write_text(json.dumps({
        "running": True,
        "timestamp": datetime.now().isoformat(),
        "pid": os.getpid()
    }))
except Exception:
    pass

try:
    while running:
        # Update timestamp every minute
        try:
            status_file.write_text(json.dumps({
                "running": True,
                "timestamp": datetime.now().isoformat(),
                "pid": os.getpid()
            }))
        except Exception:
            pass
        time.sleep(60)
except KeyboardInterrupt:
    pass

engine.stop()
# Mark as stopped in status file
try:
    status_file.write_text(json.dumps({
        "running": False,
        "timestamp": datetime.now().isoformat(),
        "pid": os.getpid()
    }))
except Exception:
    pass
print("👋 监控已停止")
