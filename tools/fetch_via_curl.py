#!/usr/bin/env python3
"""
用 curl 命令拉取 (绕过 Python HTTP 层)
"""
import sys, json, subprocess, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import DATA_DIR

def fetch_via_curl(market_filter: str):
    """用系统 curl 拉取, 绕过 Python SSL"""
    url = (
        "https://push2.eastmoney.com/api/qt/clist/get"
        f"?pn=1&pz=10000&po=1&np=1&fltt=2&fid=f3"
        f"&fs={market_filter}"
        "&fields=f12,f14"
    )
    result = subprocess.run(
        ["curl", "-s", "-k", "-m", "20", url],
        capture_output=True, text=True, timeout=25
    )
    data = json.loads(result.stdout)
    items = data.get("data", {}).get("diff", [])
    total = data.get("data", {}).get("total", 0)
    return [(it["f12"], it["f14"]) for it in items], total

A_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
HK_FS = "m:128+t:3,m:128+t:4,m:128+t:1,m:128+t:2"
US_STOCKS = json.loads(subprocess.run(
    ["python3", "-c",
     "from src.data.stock_db import US_STOCKS; import json; print(json.dumps(US_STOCKS))"],
    capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent)).stdout)

print("🔍 curl 拉取全量...")

a_stocks, a_total = fetch_via_curl(A_FS)
print(f"   A股: {len(a_stocks)} 只 (共{a_total})")

hk_stocks, hk_total = fetch_via_curl(HK_FS)
print(f"   港股: {len(hk_stocks)} 只 (共{hk_total})")

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
print(f"   A{a_len}只 HK{hk_len}只 US{len(US_STOCKS)}只".replace("a_len","A股"+str(len(a_dedup))).replace("hk_len","港股"+str(len(hk_dedup))))
