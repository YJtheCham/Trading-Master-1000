#!/usr/bin/env python3
"""
拉取全量 A股+港股 — request + verify=False
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import DATA_DIR

A_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
HK_FS = "m:128+t:3,m:128+t:4,m:128+t:1,m:128+t:2"

def fetch_all(market_filter: str, label: str):
    import urllib.request, ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    all_items = []
    for page in range(1, 30):
        url = ("https://push2.eastmoney.com/api/qt/clist/get"
               f"?pn={page}&pz=100&po=1&np=1&fltt=2&fid=f3"
               f"&fs={market_filter}&fields=f12,f14")
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Referer": "https://quote.eastmoney.com/",
        })
        try:
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"   ⚠️ {label} 第{page}页失败: {e}", flush=True)
            time.sleep(1)
            continue
        items = data.get("data", {}).get("diff", [])
        if not items:
            break
        all_items.extend([(it["f12"], it["f14"]) for it in items])
        total = data.get("data", {}).get("total", 0)
        if page == 1:
            print(f"   {label}: 共{total}只, 拉取中...", flush=True)
        if page % 10 == 0:
            print(f"   {label}: 已拉{len(all_items)}只...", flush=True)
        if len(all_items) >= total - 10:
            break
    return all_items

def main():
    print("🔍 拉取全量...", flush=True)
    a = fetch_all(A_FS, "A股")
    hk = fetch_all(HK_FS, "港股")

    from src.data.stock_db import US_STOCKS
    print(f"   美股: {len(US_STOCKS)} 只 (内置)")

    seen = set()
    a_out = [[c, n] for c, n in a if not (c in seen or seen.add(c))]
    seen.clear()
    hk_out = [[c, n] for c, n in hk if not (c in seen or seen.add(c))]

    DB = DATA_DIR / "stock_db_cache.json"
    json.dump({"A": a_out, "HK": hk_out}, open(DB, "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"\n💾 A股{len(a_out)}只 港股{len(hk_out)}只 美股{len(US_STOCKS)}只")

if __name__ == "__main__":
    main()
