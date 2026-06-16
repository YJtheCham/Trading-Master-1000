import json, os
import tushare as ts
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
df = pro.stock_basic(exchange="", list_status="L", fields="symbol,name")
a_stocks = [[r["symbol"], r["name"]] for _, r in df.iterrows()]

# 保存
Path("data/a_stocks_full.json").write_text(
    json.dumps(a_stocks, ensure_ascii=False))

from src.data.stock_db import COMMON_HK
h = [[c, n] for c, n in COMMON_HK]
Path("data/stock_db_cache.json").write_text(
    json.dumps({"A": a_stocks, "HK": h}, ensure_ascii=False))

print(f"✅ A股{len(a_stocks)}只 已保存")
