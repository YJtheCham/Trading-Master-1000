#!/usr/bin/env python3
"""极简版: 沿用首次成功的 URL 格式, 逐页拉取"""
import sys, json, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import DATA_DIR

def pull(fs: str):
    all_items = []
    for page in range(1, 60):
        url = (f"https://push2.eastmoney.com/api/qt/clist/get"
               f"?pn={page}&pz=500&po=1&np=1&fltt=2&fid=f3"
               f"&fs={fs}&fields=f12,f14")
        r = subprocess.run(["curl", "-k", "-m", "10", url],
                           capture_output=True, text=True, timeout=12)
        if not r.stdout.strip():
            break
        try: d = json.loads(r.stdout)
        except: break
        items = d.get("data",{}).get("diff",[])
        if not items: break
        all_items += [(i["f12"], i["f14"]) for i in items]
        if page == 1: print(f"  共{d['data']['total']}只, 拉取中...", flush=True)
        if page % 10 == 0: print(f"  已{len(all_items)}只", flush=True)
    return all_items

print("A股...", flush=True)
a = pull("m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23")
print("港股...", flush=True)
hk = pull("m:128+t:3,m:128+t:4,m:128+t:1,m:128+t:2")
print(f"A股{len(a)}只 港股{len(hk)}只", flush=True)

seen = set()
a_out = [[c,n] for c,n in a if not (c in seen or seen.add(c))]
seen.clear()
hk_out = [[c,n] for c,n in hk if not (c in seen or seen.add(c))]

DB = DATA_DIR / "stock_db_cache.json"
json.dump({"A":a_out, "HK":hk_out}, open(DB,"w",encoding="utf-8"), ensure_ascii=False)
print(f"💾 {DB}")
