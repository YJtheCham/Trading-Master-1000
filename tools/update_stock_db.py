#!/usr/bin/env python3
"""
更新股票离线数据库 — 全量拉取 A 股 + 港股
运行: python3 tools/update_stock_db.py
"""
import sys, time, socket, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import DATA_DIR
from src.data.stock_db import DB_CACHE_FILE, COMMON_A, COMMON_HK, US_STOCKS

socket.setdefaulttimeout(30)

def fetch_a_stocks():
    for attempt in range(3):
        try:
            print(f"🔍 拉取 A 股全量 (第{attempt+1}次)...", flush=True)
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            if df is not None and len(df) > 1000:
                return [(str(r["代码"]), str(r["名称"])) for _, r in df.iterrows()]
            print(f"   ⚠️ 数据太少 ({len(df) if df is not None else 0} 条), 重试...")
        except Exception as e:
            print(f"   ⚠️ {e}", flush=True)
            if attempt < 2:
                print("   等待 3s 重试...")
                time.sleep(3)
    return None

def fetch_hk_stocks():
    for attempt in range(3):
        try:
            print(f"🔍 拉取港股全量 (第{attempt+1}次)...", flush=True)
            import akshare as ak
            df = ak.stock_hk_spot_em()
            if df is not None and len(df) > 100:
                return [(str(r["代码"]), str(r["名称"])) for _, r in df.iterrows()]
        except Exception as e:
            print(f"   ⚠️ {e}", flush=True)
            if attempt < 2:
                time.sleep(3)
    return None

a_stocks = fetch_a_stocks()
if a_stocks:
    seen = set()
    a_dedup = [[c, n] for c, n in a_stocks if not (c in seen or seen.add(c))]
    print(f"   ✅ A 股: {len(a_dedup)} 只")
else:
    print(f"   ❌ A 股拉取 3 次均失败, 使用内置离线列表")
    a_dedup = [[c, n] for c, n in COMMON_A]

hk_stocks = fetch_hk_stocks()
if hk_stocks:
    seen = set()
    hk_dedup = [[c, n] for c, n in hk_stocks if not (c in seen or seen.add(c))]
    print(f"   ✅ 港股: {len(hk_dedup)} 只")
else:
    print(f"   ⚠️ 港股拉取失败, 使用内置列表")
    hk_dedup = [[c, n] for c, n in COMMON_HK]

cache = {"A": a_dedup, "HK": hk_dedup}
DB_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False))
print(f"\n💾 已保存到 {DB_CACHE_FILE}")
print(f"   A 股: {len(a_dedup)} 只 | 港股: {len(hk_dedup)} 只 | 美股: {len(US_STOCKS)} 只")
print(f"   缓存 24 小时有效, 之后自动刷新")
