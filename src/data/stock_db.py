"""
股票数据库: 模糊搜索 + 代码查询 + 本地缓存

用法:
  search("茅台")     → [("600519", "贵州茅台", "A"), ...]
  search("00700")    → [("00700", "腾讯控股", "HK")]
  search("AAPL")     → [("AAPL", "Apple", "US")]
"""
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from src.utils.config import DATA_DIR

DB_CACHE_FILE = DATA_DIR / "stock_db_cache.json"
CACHE_TTL = timedelta(hours=24)


# ─── 美股常用列表 (离线可用) ──────────────────────────────
# ─── 常用 A 股 (离线回退) ────────────────────────────────
COMMON_A = [
    # 白酒食品
    ("600519","贵州茅台"),("000858","五粮液"),("000568","泸州老窖"),
    ("600809","山西汾酒"),("002304","洋河股份"),("000799","酒鬼酒"),
    ("600887","伊利股份"),("002714","牧原股份"),("300498","温氏股份"),
    ("600600","青岛啤酒"),("000895","双汇发展"),("002557","洽洽食品"),
    ("603288","海天味业"),("600305","恒顺醋业"),
    # 金融
    ("601398","工商银行"),("601939","建设银行"),("601988","中国银行"),
    ("601288","农业银行"),("600036","招商银行"),("601166","兴业银行"),
    ("000001","平安银行"),("002142","宁波银行"),("601328","交通银行"),
    ("601318","中国平安"),("601628","中国人寿"),("601601","中国太保"),
    ("600030","中信证券"),("601211","国泰君安"),("600837","海通证券"),
    ("300059","东方财富"),("600570","恒生电子"),("603259","药明康德"),
    # 科技电子
    ("002415","海康威视"),("002230","科大讯飞"),("002475","立讯精密"),
    ("002916","深南电路"),("603501","韦尔股份"),("300661","圣邦股份"),
    ("002049","紫光国微"),("000725","京东方A"),("002456","欧菲光"),
    ("688981","中芯国际"),("688041","海光信息"),("688256","寒武纪"),
    ("601138","工业富联"),("002236","大华股份"),("603160","汇顶科技"),
    # 新能源
    ("300750","宁德时代"),("002594","比亚迪"),("601012","隆基绿能"),
    ("300274","阳光电源"),("002459","晶澳科技"),("688599","天合光能"),
    ("601615","明阳智能"),("300763","锦浪科技"),("600438","通威股份"),
    ("300014","亿纬锂能"),("002074","国轩高科"),("002460","赣锋锂业"),
    # 医药
    ("600276","恒瑞医药"),("300015","爱尔眼科"),("000963","华东医药"),
    ("600196","复星医药"),("002007","华兰生物"),("300122","智飞生物"),
    ("000538","云南白药"),("600436","片仔癀"),("000661","长春高新"),
    ("300760","迈瑞医疗"),("002821","凯莱英"),("300347","泰格医药"),
    # 汽车家电
    ("000625","长安汽车"),("601238","广汽集团"),("600104","上汽集团"),
    ("601966","玲珑轮胎"),("000333","美的集团"),("000651","格力电器"),
    ("600690","海尔智家"),("002050","三花智控"),("300124","汇川技术"),
    # 能源资源
    ("601857","中国石油"),("600028","中国石化"),("601088","中国神华"),
    ("600900","长江电力"),("601899","紫金矿业"),("600585","海螺水泥"),
    ("000002","万科A"),("001979","招商蛇口"),("600048","保利发展"),
    ("600809","山西汾酒"),("600585","海螺水泥"),("600031","三一重工"),
    # 通信运营商
    ("600941","中国移动"),("601728","中国电信"),("600050","中国联通"),
    # 军工物流
    ("002352","顺丰控股"),("002120","韵达股份"),("601111","中国国航"),
    ("600893","航发动力"),("000768","中航西飞"),("600760","中航沈飞"),
    # 半导体设备
    ("002371","北方华创"),("688012","中微公司"),("300782","卓胜微"),
    # 软件
    ("600536","中国软件"),("300454","深信服"),("002410","广联达"),
    ("688111","金山办公"),("300033","同花顺"),
    # 医药CXO
    ("300759","康龙化成"),("603456","九洲药业"),("688076","诺泰生物"),
    # 光伏
    ("688599","天合光能"),("002129","TCL中环"),("300316","晶盛机电"),
    # 锂电
    ("002812","恩捷股份"),("300450","先导智能"),("688005","容百科技"),
    # 消费电子
    ("601231","环旭电子"),("603986","兆易创新"),("002273","水晶光电"),
    # 特高压电力设备
    ("601877","正泰电器"),("600406","国电南瑞"),("300274","阳光电源"),
    # 化工
    ("600309","万华化学"),("002601","龙佰集团"),("600346","恒力石化"),
    # 更多常见股
    ("603993","洛阳钼业"),("000568","泸州老窖"),("600809","山西汾酒"),
    ("000623","吉林敖东"),("002001","新和成"),("600741","华域汽车"),
    ("603899","晨光文具"),("002508","老板电器"),("000100","TCL科技"),
    ("300413","芒果超媒"),("002624","完美世界"),
    ("688396","华润微"),
]

# ─── 港股常用 (离线回退) ─────────────────────────────────
COMMON_HK = [
    ("00700", "腾讯控股"), ("09988", "阿里巴巴"), ("01810", "小米集团"),
    ("03690", "美团"), ("09618", "京东"), ("09888", "百度"),
    ("00883", "中国海洋石油"), ("00941", "中国移动"),
    ("01347", "华虹半导体"), ("02015", "理想汽车"),
    ("00388", "香港交易所"), ("00005", "汇丰控股"),
    ("01299", "友邦保险"), ("02318", "中国平安"),
    ("02269", "药明生物"), ("06160", "百济神州"),
]

# ─── 美股常用列表 (离线可用) ──────────────────────────────
US_STOCKS = [
    ("AAPL", "Apple"), ("GOOGL", "Alphabet"), ("GOOG", "Alphabet C"),
    ("MSFT", "Microsoft"), ("AMZN", "Amazon"), ("META", "Meta"),
    ("NVDA", "NVIDIA"), ("TSLA", "Tesla"), ("BRK.B", "Berkshire Hathaway"),
    ("JPM", "JPMorgan Chase"), ("V", "Visa"), ("JNJ", "Johnson & Johnson"),
    ("WMT", "Walmart"), ("MA", "Mastercard"), ("PG", "Procter & Gamble"),
    ("UNH", "UnitedHealth"), ("HD", "Home Depot"), ("DIS", "Disney"),
    ("BAC", "Bank of America"), ("NFLX", "Netflix"), ("ADBE", "Adobe"),
    ("CRM", "Salesforce"), ("PYPL", "PayPal"), ("INTC", "Intel"),
    ("AMD", "AMD"), ("CSCO", "Cisco"), ("CMCSA", "Comcast"),
    ("PEP", "PepsiCo"), ("KO", "Coca-Cola"), ("ABNB", "Airbnb"),
    ("UBER", "Uber"), ("SQ", "Block"), ("SNAP", "Snap"),
    ("SPY", "SPDR S&P 500"), ("QQQ", "Invesco QQQ"),
    ("DIA", "SPDR Dow Jones"), ("IWM", "Russell 2000"),
    ("TLT", "20+ Year Treasury"), ("GLD", "SPDR Gold"),
    ("SLV", "iShares Silver"), ("XLE", "Energy Sector"),
    ("XLF", "Financial Sector"), ("XLK", "Technology Sector"),
    ("BABA", "Alibaba"), ("JD", "JD.com"), ("PDD", "Pinduoduo"),
    ("NIO", "NIO"), ("LI", "Li Auto"), ("XPEV", "XPeng"),
    ("BIDU", "Baidu"), ("TCEHY", "Tencent (ADR)"),
]

# ─── 股票数据库 ────────────────────────────────────────────
class StockDatabase:
    def __init__(self):
        self._data: dict[str, list[tuple[str, str]]] = {
            "A": [], "HK": [], "US": US_STOCKS.copy(),
        }
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self._loaded = True

        # 先用离线列表兜底 (去重)
        seen_a = set()
        dedup_a = []
        for code, name in COMMON_A:
            if code not in seen_a:
                seen_a.add(code)
                dedup_a.append((code, name))
        self._data["A"] = dedup_a
        dedup_hk = []
        seen_hk = set()
        for code, name in COMMON_HK:
            if code not in seen_hk:
                seen_hk.add(code)
                dedup_hk.append((code, name))
        self._data["HK"] = dedup_hk

        # 尝试用缓存 (要求至少 500 只 A 股, 否则视为不完整)
        if DB_CACHE_FILE.exists():
            age = datetime.now() - datetime.fromtimestamp(DB_CACHE_FILE.stat().st_mtime)
            if age < CACHE_TTL:
                try:
                    cached = json.loads(DB_CACHE_FILE.read_text())
                    ca = cached.get("A", [])
                    if len(ca) >= 500:
                        self._data["A"] = ca
                    if cached.get("HK"):
                        self._data["HK"] = cached["HK"]
                    if len(ca) >= 500:
                        return
                except Exception:
                    pass

        # 尝试从直接拉取的全量文件加载
        full_file = DATA_DIR / "a_stocks_full.json"
        if full_file.exists():
            try:
                self._data["A"] = json.loads(full_file.read_text())
                return
            except Exception:
                pass

        # 从网络加载更全的列表 (超时5秒)
        import concurrent.futures

        def _load_a():
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            self._data["A"] = [(str(r["代码"]), str(r["名称"])) for _, r in df.iterrows()]

        def _load_hk():
            import akshare as ak
            df = ak.stock_hk_spot_em()
            self._data["HK"] = [(str(r["代码"]), str(r["名称"])) for _, r in df.iterrows()]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = {pool.submit(_load_a): "A", pool.submit(_load_hk): "HK"}
            concurrent.futures.wait(futs, timeout=5)

        # 缓存
        try:
            DB_CACHE_FILE.write_text(json.dumps(self._data, ensure_ascii=False))
        except Exception:
            pass

    def search(self, query: str, market: Optional[str] = None,
               limit: int = 20) -> list[tuple[str, str, str]]:
        """模糊搜索: 按代码或名称匹配"""
        self._load()
        results = []
        q = query.strip().lower()
        if not q:
            return results

        markets = [market] if market else ["A", "HK", "US"]
        for m in markets:
            for code, name in self._data.get(m, []):
                if q in code.lower() or q in name.lower():
                    results.append((code, name, m))
                    if len(results) >= limit:
                        return results
        return results

    def get_name(self, code: str, market: str) -> str:
        """根据代码和市查找名称"""
        self._load()
        for c, name in self._data.get(market, []):
            if c == code:
                return name
        # 美股特殊处理: 可能有重复
        if market == "US":
            for c, name in US_STOCKS:
                if c == code:
                    return name
        return ""

    def all_stocks(self, market: str) -> list[tuple[str, str]]:
        self._load()
        return self._data.get(market, [])


# ─── 单例 ─────────────────────────────────────────────────
_db: Optional[StockDatabase] = None

def get_db() -> StockDatabase:
    global _db
    if _db is None:
        _db = StockDatabase()
    return _db


def search_stocks(query: str, market: Optional[str] = None, limit: int = 20):
    return get_db().search(query, market, limit)


def get_stock_name(code: str, market: str) -> str:
    return get_db().get_name(code, market)


def resolve_stock_name(code: str, market: str) -> str:
    """获取股票名称：本地库 → 联网反查 → 更新缓存"""
    name = get_db().get_name(code, market)
    if name:
        return name
    try:
        if market == "A":
            name = _fetch_a_name(code)
        elif market == "US":
            name = _fetch_us_name(code)
        elif market == "HK":
            name = _fetch_a_name(code) or _fetch_us_name(code)
        if name:
            # 更新本地缓存, 下次就快了
            get_db()._data[market].append((code, name))
    except Exception:
        pass
    return name


def _fetch_a_name(code: str) -> str:
    """akshare 单只 A 股名称 (全量拉取备缓存, 超时 10s)"""
    import threading, akshare as ak
    result = [""]
    def _run():
        try:
            df = ak.stock_zh_a_spot_em()
            rows = [(str(r["代码"]), str(r["名称"])) for _, r in df.iterrows()]
            for c, n in rows:
                if c == code:
                    result[0] = n
            # 顺便更新整个数据库
            if rows:
                get_db()._data["A"] = rows
        except Exception:
            pass
    t = threading.Thread(target=_run)
    t.start()
    t.join(timeout=10)
    return result[0]


def _fetch_us_name(code: str) -> str:
    """yfinance 获取美股名称"""
    try:
        import yfinance as yf
        t = yf.Ticker(code)
        info = t.fast_info
        return str(info.get("shortName") or info.get("longName") or "")
    except Exception:
        return ""


# ─── 文件导入 ─────────────────────────────────────────────
def parse_stock_file(filepath: str) -> list[tuple[str, str, str]]:
    """解析 CSV/TXT/XLSX 文件, 返回 [(code, name, market), ...]

    自动检测格式:
      - CSV/XLSX: 查找 code/symbol/代码 列
      - TXT: 每行一个代码, 或 code,name 格式
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    suffix = path.suffix.lower()
    codes: list[tuple[str, str, str]] = []

    if suffix == ".txt":
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            # 尝试解析 "code,name,market" 或 "code" 格式
            parts = [p.strip() for p in line.split(",")]
            code = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            market = parts[2] if len(parts) > 2 else _guess_market(code)
            codes.append((code, name, market))

    elif suffix in (".csv", ".xlsx"):
        import pandas as pd
        df = pd.read_excel(filepath) if suffix == ".xlsx" else pd.read_csv(filepath)
        # 找代码列
        code_col = None
        name_col = None
        market_col = None
        for col in df.columns:
            cl = col.lower()
            if any(k in cl for k in ["代码", "代码", "code", "symbol", "ts_code", "stock_code"]):
                code_col = col
            elif any(k in cl for k in ["名称", "name", "stock_name"]):
                name_col = col
            elif any(k in cl for k in ["市场", "market", "交易所"]):
                market_col = col

        if code_col is None:
            # 尝试第一列
            code_col = df.columns[0]

        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            if not code or code == "nan":
                continue
            name = str(row[name_col]).strip() if name_col and name_col in row else ""
            market = str(row[market_col]).strip().upper() if market_col and market_col in row else _guess_market(code)
            codes.append((code, name, market))
    else:
        raise ValueError(f"不支持的文件格式: {suffix} (支持: .csv, .txt, .xlsx)")

    return codes


def _guess_market(code: str) -> str:
    """根据代码格式猜测市场"""
    code = code.strip().upper()
    if code.endswith(".HK"):
        return "HK"
    elif code.endswith(".SS") or code.endswith(".SZ"):
        return "A"
    elif len(code) <= 5 and not code.startswith(("6", "0", "3")):
        return "US"
    elif code.startswith(("6", "9")):
        return "A"
    elif code.startswith(("0", "3")):
        return "A"
    elif code.startswith(("00", "01", "02", "03", "05", "08")):
        # 港股常见开头
        return "HK" if code[0] == "0" else "US"
    return "US"
