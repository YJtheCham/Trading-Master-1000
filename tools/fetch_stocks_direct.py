#!/usr/bin/env python3
"""
直接 HTTP 拉取全量 A 股代码 → 写成本地文件
无需 akshare, 绕过 Python 3.14 HTTP 层 bug
运行: python3 tools/fetch_stocks_direct.py
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import DATA_DIR

OUTPUT = DATA_DIR / "a_stocks_full.json"

def fetch_via_jsonp(page=1, pz=5000):
    """通过东方财富 HTTP API 拉取"""
    import urllib.request
    url = (
        f"https://push2.eastmoney.com/api/qt/clist/get"
        f"?pn={page}&pz={pz}&po=1&np=1&fltt=2&fid=f3"
        f"&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
        f"&fields=f12,f14,f2,f3,f4,f15,f16,f17,f18"
    )
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Referer": "https://quote.eastmoney.com/",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    items = data.get("data", {}).get("diff", [])
    total = data.get("data", {}).get("total", 0)
    return [(it["f12"], it["f14"]) for it in items], total

print("🔍 从东方财富 API 拉取全量 A 股...", flush=True)

for attempt in range(3):
    try:
        stocks, total = fetch_via_jsonp()
        if len(stocks) > 1000:
            print(f"   ✅ 第{attempt+1}次成功: {len(stocks)} 只 (服务器共{total}只)")

            # 保存
            with open(OUTPUT, "w", encoding="utf-8") as f:
                json.dump(stocks, f, ensure_ascii=False)
            print(f"💾 已保存到 {OUTPUT}")
            sys.exit(0)
        else:
            print(f"   ⚠️ 数据太少 ({len(stocks)}条), 重试...")
    except Exception as e:
        print(f"   ⚠️ 第{attempt+1}次失败: {e}", flush=True)
        time.sleep(2)

print("❌ 3 次重试均失败")
sys.exit(1)
