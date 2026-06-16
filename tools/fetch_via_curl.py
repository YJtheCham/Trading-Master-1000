#!/usr/bin/env python3
"""
用 curl 命令拉取 (绕过 Python HTTP 层)
"""
import sys, json, subprocess, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import DATA_DIR

def fetch_via_curl(market_filter: str, label: str):
    """用系统 curl 拉取所有分页, 绕过 Python SSL"""
    all_items = []
    for page in range(1, 20):
        url = (
            "https://push2.eastmoney.com/api/qt/clist/get"
            f"?pn={page}&pz=500&po=1&np=1&fltt=2&fid=f3"
            f"&fs={market_filter}"
            "&fields=f12,f14"
        )
        result = subprocess.run(
            ["curl", "-s", "-k", "-m", "15", url],
            capture_output=True, text=True, timeout=20
        )
        if not result.stdout.strip():
            break
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            break
        items = data.get("data", {}).get("diff", [])
        if not items:
            break
        all_items.extend([(it["f12"], it["f14"]) for it in items])
        total = data.get("data", {}).get("total", 0)
        if page == 1:
            print(f"   {label}: 共{total}只, 正在拉取...", flush=True)
        if page * 500 >= total:
            break
    return all_items

A_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
HK_FS = "m:128+t:3,m:128+t:4,m:128+t:1,m:128+t:2"
US_STOCKS = json.loads(subprocess.run(
    ["python3", "-c",
     "from src.data.stock_db import US_STOCKS; import json; print(json.dumps(US_STOCKS))"],
    capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent)).stdout)

print("🔍 curl 拉取全量...")

a_stocks = fetch_via_curl(A_FS, "A股")
hk_stocks = fetch_via_curl(HK_FS, "港股")

print(f"   美股: {len(US_STOCKS)} 只 (内置)")

# 去重
seen = set()
a_dedup = [[c,n] for c,n in a_stocks if not (c in seen or seen.add(c))]
seen.clear()
hk_dedup = [[c,n] for c,n in hk_stocks if not (c in seen or seen.add(c))]

cache = {"A": a_dedup, "HK": hk_dedup}
DB = DATA_DIR / "stock_db_cache.json"
json.dump(cache, open(DB, "w", encoding="utf-8"), ensure_ascii=False)

print(f"\n💾 {DB}")
print(f"   A股{len(a_dedup)}只 港股{len(hk_dedup)}只 美股{len(US_STOCKS)}只")
