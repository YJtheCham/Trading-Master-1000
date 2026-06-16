#!/usr/bin/env python3
"""
直接 HTTP 拉取全量 A 股 + 港股 → 写入本地文件
无需 akshare, 绕过 Python 3.14 HTTP 层 bug
运行: python3 tools/fetch_stocks_direct.py
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import DATA_DIR

def fetch_eastmoney(market_filter: str, label: str):
    """东方财富 API 通用拉取"""
    import urllib.request, ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    url = (
        f"https://push2.eastmoney.com/api/qt/clist/get"
        f"?pn=1&pz=10000&po=1&np=1&fltt=2&fid=f3"
        f"&fs={market_filter}"
        f"&fields=f12,f14"
    )
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Referer": "https://quote.eastmoney.com/",
    })
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        data = json.loads(resp.read().decode())
    items = data.get("data", {}).get("diff", [])
    total = data.get("data", {}).get("total", 0)
    stocks = [(it["f12"], it["f14"]) for it in items]
    print(f"   ✅ {label}: {len(stocks)} 只 (服务器共{total}只)", flush=True)
    return stocks

def fetch_with_retry(market_filter: str, label: str, min_count=100):
    for attempt in range(3):
        try:
            stocks = fetch_eastmoney(market_filter, label)
            if len(stocks) >= min_count:
                return stocks
            print(f"   ⚠️ 数据太少, 重试...", flush=True)
        except Exception as e:
            print(f"   ⚠️ 第{attempt+1}次失败: {e}", flush=True)
        time.sleep(2)
    return []

# ─── A 股 ─────────────────────────────────────────────────
A_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
# ─── 港股 ─────────────────────────────────────────────────
HK_FS = "m:128+t:3,m:128+t:4,m:128+t:1,m:128+t:2"

print("🔍 拉取全量股票数据...", flush=True)

a_stocks = fetch_with_retry(A_FS, "A股", 1000)
hk_stocks = fetch_with_retry(HK_FS, "港股", 100)

# ─── 美股: 从内置列表 ────────────────────────────────────
us_stocks = [
    ["AAPL","Apple"],["MSFT","Microsoft"],["GOOGL","Alphabet"],
    ["AMZN","Amazon"],["NVDA","NVIDIA"],["META","Meta"],
    ["TSLA","Tesla"],["BRK.B","Berkshire Hathaway"],
    ["JPM","JPMorgan"],["V","Visa"],["JNJ","J&J"],
    ["WMT","Walmart"],["MA","Mastercard"],["PG","P&G"],
    ["UNH","UnitedHealth"],["HD","Home Depot"],["DIS","Disney"],
    ["BAC","Bank of America"],["NFLX","Netflix"],["ADBE","Adobe"],
    ["CRM","Salesforce"],["PYPL","PayPal"],["INTC","Intel"],
    ["AMD","AMD"],["CSCO","Cisco"],["PEP","PepsiCo"],["KO","Coca-Cola"],
    ["ABNB","Airbnb"],["UBER","Uber"],["SQ","Block"],["SNAP","Snap"],
    ["SPY","S&P500 ETF"],["QQQ","Nasdaq ETF"],["DIA","Dow ETF"],
    ["IWM","Russell2000"],["TLT","Treasury ETF"],["GLD","Gold ETF"],
    ["SLV","Silver ETF"],["XLE","Energy ETF"],["XLF","Finance ETF"],
    ["XLK","Tech ETF"],["BABA","Alibaba"],["JD","JD.com"],
    ["PDD","Pinduoduo"],["NIO","NIO"],["LI","Li Auto"],
    ["XPEV","XPeng"],["BIDU","Baidu"],["TCEHY","Tencent ADR"],
    ["PLTR","Palantir"],["SNOW","Snowflake"],["DDOG","Datadog"],
    ["CRWD","CrowdStrike"],["ZS","Zscaler"],["NET","Cloudflare"],
    ["MDB","MongoDB"],["SHOP","Shopify"],["COIN","Coinbase"],
    ["RBLX","Roblox"],["U","Unity"],["AFRM","Affirm"],
    ["SOFI","SoFi"],["HOOD","Robinhood"],["RIVN","Rivian"],
    ["LCID","Lucid"],["F","Ford"],["GM","GM"],["BA","Boeing"],
    ["CAT","Caterpillar"],["DE","Deere"],["XOM","ExxonMobil"],
    ["CVX","Chevron"],["OXY","Occidental"],["COP","ConocoPhillips"],
    ["NEE","NextEra Energy"],["ENPH","Enphase"],["FSLR","First Solar"],
    ["PFE","Pfizer"],["MRNA","Moderna"],["BNTX","BioNTech"],
    ["LLY","Eli Lilly"],["UNH","UnitedHealth"],["ABBV","AbbVie"],
    ["COST","Costco"],["TGT","Target"],["LOW","Lowes"],
    ["NKE","Nike"],["SBUX","Starbucks"],["MCD","McDonalds"],
    ["ORCL","Oracle"],["IBM","IBM"],["CRM","Salesforce"],
    ["NOW","ServiceNow"],["WDAY","Workday"],["TEAM","Atlassian"],
    ["AVGO","Broadcom"],["QCOM","Qualcomm"],["TXN","Texas Instruments"],
    ["MU","Micron"],["ARM","ARM Holdings"],["MRVL","Marvell"],
    ["SMCI","Super Micro"],["DELL","Dell"],["HPQ","HP"],
]
print(f"   ✅ 美股: {len(us_stocks)} 只 (内置常用列表)", flush=True)

# ─── 去重 + 保存 ─────────────────────────────────────────
seen_a = set()
a_dedup = []
for code, name in a_stocks:
    if code not in seen_a:
        seen_a.add(code)
        a_dedup.append([code, name])

seen_hk = set()
hk_dedup = []
for code, name in hk_stocks:
    if code not in seen_hk:
        seen_hk.add(code)
        hk_dedup.append([code, name])

cache = {"A": a_dedup, "HK": hk_dedup}

# 保存到缓存文件 (stock_db 读取的)
DB_CACHE = DATA_DIR / "stock_db_cache.json"
json.dump(cache, open(DB_CACHE, "w", encoding="utf-8"), ensure_ascii=False)

# 也保存 A 股单独文件
json.dump(a_dedup, open(DATA_DIR / "a_stocks_full.json", "w", encoding="utf-8"), ensure_ascii=False)

print(f"\n💾 已保存: A股{len(a_dedup)}只 港股{len(hk_dedup)}只 美股{len(us_stocks)}只")
print(f"   缓存: {DB_CACHE}")
sys.exit(0)
