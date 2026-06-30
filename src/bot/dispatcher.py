"""
Agent 任务分发器: 解析群组消息 → 调用对应 agent → 回复结果
"""
import json
import logging
import re
import time
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TASK_LOG = Path(__file__).resolve().parent.parent.parent / "data" / "bot_tasks.json"


@dataclass
class AgentTask:
    task_id: str = ""
    source: str = ""
    group_name: str = ""
    sender: str = ""
    message: str = ""
    agent: str = ""
    command: str = ""
    status: str = "pending"
    result: str = ""
    created_at: str = ""
    completed_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")
        if not self.task_id:
            self.task_id = f"task_{int(time.time())}"


AGENT_MAP = {
    "pm": {
        "name": "PM Agent",
        "keywords": ["pm", "产品", "review", "审查", "测试", "bug", "ux", "审计", "检查app"],
        "description": "系统性审查所有功能模块，输出结构化报告",
        "agent_file": "stock-pm",
    },
    "trader": {
        "name": "Trader Agent",
        "keywords": ["trader", "交易", "分析", "策略", "提醒", "每日报告", "自选", "选股", "监控", "复盘"],
        "description": "分析自选股、推荐策略、设置提醒、生成每日报告",
        "agent_file": "stock-trader",
    },
}

COMMAND_MAP = {
    "review": {"agent": "pm", "prompt": "系统性review所有10个功能模块，验证数据正确性、UI交互、边界情况、移动端体验、性能，输出结构化报告"},
    "audit": {"agent": "pm", "prompt": "审计app的所有模块，找出致命问题、Bug、UX改进、功能缺口"},
    "test": {"agent": "pm", "prompt": "测试所有功能点，验证数据正确性、交互功能、边界情况"},
    "daily": {"agent": "trader", "prompt": "执行每日交易分析完整流程：盘面扫描→个股分析→提醒设置→策略推荐→生成每日报告"},
    "analyze": {"agent": "trader", "prompt": "分析自选股，评估每只股票的技术面、基本面、模型预测，推荐策略和操作建议"},
    "alerts": {"agent": "trader", "prompt": "检查并更新所有自选股的交易提醒，添加缺失的关键监控规则"},
    "strategy": {"agent": "trader", "prompt": "为自选股推荐最优交易策略，基于模型预测和回测结果"},
    "report": {"agent": "trader", "prompt": "生成今日交易分析报告，包含盘面概况、自选扫描、重点个股、提醒状态、App问题发现"},
}


DIRECT_ACTION_COMMANDS = {
    "add": "action",
    "remove": "action",
    "watchlist": "action",
    "alert": "action",
    "strategy": "action",
    "predict": "action",
    "scan": "action",
    "wind_query": "action",
    "wind_search": "action",
}


def _execute_direct_action(command: str, args: str) -> str:
    """直接执行 StockPredict 操作，不走 agent/LLM"""
    cmd = command.lower()

    if cmd == "add":
        return _action_add_stock(args)
    elif cmd == "remove":
        return _action_remove_stock(args)
    elif cmd == "watchlist":
        return _action_watchlist(args)
    elif cmd == "alert":
        return _action_alert(args)
    elif cmd == "strategy":
        return _action_strategy(args)
    elif cmd == "predict":
        return _action_predict(args)
    elif cmd == "scan":
        return _action_scan(args)
    return f"未知操作: {cmd}"


def _action_add_stock(args: str) -> str:
    """添加自选股: /add 600519:A:贵州茅台 或 /add 600519 A 茅台"""
    from src.utils.config import StockItem
    from src.data.fetcher import load_watchlist, save_watchlist

    parsed = _parse_stock_args(args)
    if not parsed:
        return "格式: /add 代码:市场:名称\n例: /add 600519:A:贵州茅台\n或: /add 600519 A 茅台"

    code, market, name = parsed
    items = load_watchlist()
    if any(i.symbol == code and i.market == market for i in items):
        return f"⚠️ {name}({code}) 已在自选列表中"
    items.append(StockItem(symbol=code, market=market, name=name))
    save_watchlist(items)
    return f"✅ 已添加 {name}({market}:{code}) 到自选列表 (共{len(items)}只)"


def _action_remove_stock(args: str) -> str:
    """删除自选股: /remove 600519 或 /remove 600519:A"""
    from src.data.fetcher import load_watchlist, save_watchlist

    code_match = re.match(r'^(\d{6})\s*[:\s]*([A-Z]*)', args.strip())
    if not code_match:
        return "格式: /remove 代码\n例: /remove 600519"

    code = code_match.group(1)
    market = code_match.group(2) or None

    items = load_watchlist()
    removed = None
    new_items = []
    for i in items:
        if i.symbol == code and (market is None or i.market == market):
            removed = i
        else:
            new_items.append(i)

    if removed:
        save_watchlist(new_items)
        return f"✅ 已移除 {removed.name}({removed.market}:{removed.symbol}) (剩余{len(new_items)}只)"
    return f"⚠️ {code} 不在自选列表中"


def _action_watchlist(args: str) -> str:
    """查看自选股列表: /watchlist"""
    from src.data.fetcher import load_watchlist, fetch_data

    items = load_watchlist()
    if not items:
        return "⚠️ 自选列表为空，使用 /add 添加股票"

    lines = []
    for i in items:
        try:
            df = fetch_data(i.symbol, i.market, period_days=5, use_cache=True)
            if not df.empty:
                price = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2]) if len(df) > 1 else price
                change = (price - prev) / prev * 100
                lines.append(f"{i.name}({i.market}:{i.symbol}) {price:.2f} {change:+.1f}%")
            else:
                lines.append(f"{i.name}({i.market}:{i.symbol}) 暂无数据")
        except Exception:
            lines.append(f"{i.name}({i.market}:{i.symbol}) 数据获取失败")

    header = f"📋 自选股列表 ({len(items)}只)\n"
    return header + "\n".join(lines)


def _action_alert(args: str) -> str:
    """交易监控: /alert add 600519:A:金叉:short=5,long=20
                 /alert list
                 /alert remove <uid>
                 /alert toggle <uid>"""
    from src.alerts import AlertRule, add_rule, load_rules, remove_rule, toggle_rule, CONDITION_TYPES

    parts = args.strip().split()
    if not parts:
        cond_list = "\n".join(f"  {k}: {v}" for k, v in CONDITION_TYPES.items())
        return f"用法:\n  /alert add 代码:市场:条件:参数\n  /alert list\n  /alert remove uid\n  /alert toggle uid\n\n可用条件:\n{cond_list}"

    sub_cmd = parts[0].lower()

    if sub_cmd == "list":
        rules = load_rules()
        if not rules:
            return "⚠️ 暂无监控规则，使用 /alert add 添加"
        lines = []
        for r in rules:
            status = "🟢" if r.enabled else "🔴"
            params_str = ", ".join(f"{k}={v}" for k, v in r.params.items()) if r.params else ""
            lines.append(f"{status} [{r.uid[:30]}] {r.symbol}({r.market}) {r.condition}({params_str}) {r.label}")
        return f"📋 监控规则 ({len(rules)}条)\n" + "\n".join(lines)

    elif sub_cmd == "add":
        # Format: /alert add 600519:A:golden_cross:short=5,long=20:茅台金叉
        detail = " ".join(parts[1:])
        segments = detail.split(":")
        if len(segments) < 3:
            return "格式: /alert add 代码:市场:条件[:参数][:标签]\n例: /alert add 600519:A:golden_cross:short=5,long=20:茅台金叉"

        symbol = segments[0]
        market = segments[1]
        condition = segments[2]
        params = {}
        label = ""

        if condition not in CONDITION_TYPES:
            return f"⚠️ 未知条件 '{condition}'，可用条件: {', '.join(CONDITION_TYPES.keys())}"

        if len(segments) > 3:
            param_str = segments[3]
            for pair in param_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    try:
                        params[k.strip()] = float(v.strip()) if "." in v else int(v.strip())
                    except ValueError:
                        params[k.strip()] = v.strip()
        if len(segments) > 4:
            label = segments[4]

        rule = AlertRule(
            symbol=symbol, market=market, condition=condition,
            params=params, label=label,
            interval_minutes=5, cooldown_minutes=60,
        )
        add_rule(rule)
        return f"✅ 已添加监控: {CONDITION_TYPES.get(condition, condition)}\n   {symbol}({market}) 参数:{params} 标签:{label}\n   UID: {rule.uid}"

    elif sub_cmd == "remove":
        uid = " ".join(parts[1:])
        if remove_rule(uid):
            return f"✅ 已移除规则 {uid}"
        return f"⚠️ 未找到规则 {uid}"

    elif sub_cmd == "toggle":
        uid = " ".join(parts[1:])
        rule = toggle_rule(uid)
        if rule:
            status = "启用" if rule.enabled else "禁用"
            return f"✅ 规则 {uid} 已切换为{status}"
        return f"⚠️ 未找到规则 {uid}"

    return f"未知子命令: {sub_cmd}"


def _action_strategy(args: str) -> str:
    from src.data.fetcher import fetch_data, load_watchlist
    from src.recommend.engine import scan_strategies

    stocks = _resolve_stocks_from_args(args)
    if not stocks:
        return "格式: /strategy 代码:市场\n例: /strategy 600519:A\n或 /strategy (全自选)"

    results = []
    for code, market, name in stocks[:5]:
        try:
            df = fetch_data(code, market, period_days=120, use_cache=True)
            if df.empty or len(df) < 30:
                results.append(f"⚠️ {name}({code}) 数据不足")
                continue
            strat_results = scan_strategies(df, code, market, capital=100000)
            valid = [s for s in strat_results if not s.error]
            if not valid:
                errors = [s.error for s in strat_results if s.error][:3]
                results.append(f"⚠️ {name}({code}) 无有效策略: {errors}")
                continue
            best = max(valid, key=lambda x: x.total_return)
            results.append(
                f"**{name}**({market}:{code})\n"
                f"  最优策略: {best.strategy} 回报={best.total_return*100:.1f}%\n"
                f"  Sharpe={best.sharpe:.2f} MaxDD={best.max_dd*100:.1f}%\n"
                f"  胜率={best.win_rate*100:.1f}% 交易次数={best.total_trades}"
            )
            top3 = sorted(valid, key=lambda x: x.total_return, reverse=True)[:3]
            for s in top3:
                emoji = "🟢" if s.total_return > 0 else "🔴"
                results.append(f"  {emoji} {s.strategy}: 回报{s.total_return*100:+.1f}% Sharpe={s.sharpe:.2f} MaxDD={s.max_dd*100:.1f}%")
        except Exception as e:
            results.append(f"⚠️ {name}({code}) 策略分析失败: {e}")

    return f"📊 策略推荐\n" + "\n".join(results)


def _action_predict(args: str) -> str:
    from src.data.fetcher import fetch_data, load_watchlist
    from src.models.factory import run_models

    stocks = _resolve_stocks_from_args(args)
    if not stocks:
        return "格式: /predict 代码:市场\n例: /predict 600519:A\n或 /predict (全自选)"

    results = []
    for code, market, name in stocks[:5]:
        try:
            df = fetch_data(code, market, period_days=120, use_cache=True)
            if df.empty or len(df) < 30:
                results.append(f"⚠️ {name}({code}) 数据不足")
                continue
            preds = run_models(df, model_names=["GBDT", "XGBoost", "LSTM"], steps=30)
            price_now = float(df["Close"].iloc[-1])

            pred_lines = []
            for model_name, pred_result in preds.items():
                if pred_result.error:
                    pred_lines.append(f"  {model_name}: 预测失败({pred_result.error})")
                elif len(pred_result.forecast) > 0:
                    pred_price = float(pred_result.forecast[-1])
                    pct = (pred_price - price_now) / price_now * 100
                    pred_lines.append(f"  {model_name}: {pred_price:.2f} ({pct:+.1f}%)")

            results.append(f"**{name}**({market}:{code}) 现价{price_now:.2f}\n" + "\n".join(pred_lines))
        except Exception as e:
            results.append(f"⚠️ {name}({code}) 预测失败: {e}")

    return f"📈 模型预测 (30天)\n" + "\n".join(results)


WIND_CLI_DIR = Path.home() / ".config" / "opencode" / "skills" / "skills" / "wind-mcp-skill"

WIND_CODE_MAP = {
    "A": lambda s: s + ".SH" if s.startswith(("6", "9")) else s + ".SZ",
    "HK": lambda s: s.zfill(5) + ".HK",
    "US": lambda s: s if "." in s else s + ".O",
}

WIND_QUERY_TOOLS = {
    "quote": ("stock_data", "get_stock_price_indicators",
              "indexes=中文简称,最新成交价,涨跌幅,成交量,成交额,最高价,最低价,开盘价,昨收盘"),
    "kline": ("stock_data", "get_stock_kline",
              ""),
    "basicinfo": ("stock_data", "get_stock_basicinfo",
              ""),
    "financial": ("stock_data", "get_stock_financial",
              ""),
}


def _call_wind_mcp(server: str, tool: str, params: dict) -> Optional[str]:
    """调用 Wind MCP CLI 获取数据"""
    cli_dir = None
    for candidate in [WIND_CLI_DIR, Path("/root/.config/opencode/skills/skills/wind-mcp-skill")]:
        if candidate.is_dir() and (candidate / "scripts" / "cli.mjs").exists():
            cli_dir = candidate
            break
    if not cli_dir:
        return None
    try:
        cmd = ["node", "scripts/cli.mjs", "call", server, tool,
               json.dumps(params, ensure_ascii=False)]
        r = subprocess.run(cmd, cwd=str(cli_dir), capture_output=True, text=True, timeout=30)
        data = json.loads(r.stdout)
        if data.get("isError"):
            return None
        text = data.get("content", [{}])[0].get("text", "{}")
        inner = json.loads(text)
        if inner.get("error"):
            return None
        return json.dumps(inner.get("data", {}), ensure_ascii=False, indent=2)[:2000]
    except Exception as e:
        logger.warning(f"Wind MCP CLI failed: {e}")
        return None


def _action_wind_query(params: dict) -> str:
    """调用 Wind MCP 获取行情/基本面数据, 再让 LLM 分析"""
    symbol = params.get("symbol", "")
    market = params.get("market", "A")
    query_type = params.get("query_type", "quote")

    if not symbol:
        return "⚠️ 请提供股票代码"

    windcode = WIND_CODE_MAP.get(market, lambda s: s)(symbol)

    server, tool, indexes = WIND_QUERY_TOOLS.get(query_type, WIND_QUERY_TOOLS["quote"])

    wind_params = {"windcode": windcode}
    if query_type == "quote" and indexes:
        wind_params["indexes"] = indexes
    elif query_type == "kline":
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime("%Y%m%d")
        begin_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        wind_params["begin_date"] = begin_date
        wind_params["end_date"] = end_date
    elif query_type == "basicinfo":
        wind_params["question"] = f"{windcode}公司基本档案"

    raw_data = _call_wind_mcp(server, tool, wind_params)

    if not raw_data:
        return f"⚠️ Wind MCP 查询 {symbol}({market}) {query_type} 失败，请检查配置"

    from src.recommend.advisor import call_llm
    from src.utils.config import get_llm_config
    cfg = get_llm_config()

    prompt = f"""请根据以下Wind MCP实时数据，简洁回答用户的问题(200字以内):
股票: {symbol}({market})
查询类型: {query_type}
数据: {raw_data}

请用中文回答，格式清晰，突出关键指标。"""

    reply = call_llm(prompt, cfg["api_key"], cfg["base_url"], cfg["model"],
                     system_prompt="你是投资数据分析师，根据Wind实时数据简洁回答，不要编造数据。")
    return reply or f"📊 Wind数据已获取，但LLM分析失败\n原始数据: {raw_data[:500]}"


def _action_skill_analysis(params: dict, skill_name: str) -> str:
    """用指定skill框架分析: 加载skill完整内容 → 注入LLM → 框架化分析"""
    topic = params.get("topic", params.get("symbol", ""))
    market = params.get("market", "A")
    symbol = params.get("symbol", "")

    if not topic:
        return "⚠️ 请提供分析主题或股票代码"

    # 加载skill完整内容
    skill_file = SKILLS_DIR / f"{skill_name}.md"
    skill_content = ""
    if skill_file.exists():
        skill_content = skill_file.read_text()
    else:
        return f"⚠️ {skill_name} 技能文件不存在"

    from src.recommend.advisor import call_llm
    from src.utils.config import get_llm_config
    cfg = get_llm_config()

    # 如果有股票代码, 先获取Wind行情数据
    data_context = ""
    if symbol:
        windcode = WIND_CODE_MAP.get(market, lambda s: s)(symbol)
        raw = _call_wind_mcp("stock_data", "get_stock_price_indicators",
                             {"windcode": windcode, "indexes": "中文简称,最新成交价,涨跌幅,成交量"})
        if raw:
            data_context = f"\n\nWind实时数据: {raw[:1500]}"

    system_prompt = f"""你是投资分析专家, 严格按照以下分析框架回答用户问题:

{skill_content}

回答要求:
1. 严格按照框架的方法论流程分析
2. 结合提供的实时数据(如有)
3. 有证据支撑的给出明确判断
4. 缺乏数据的指明确认路径
5. 用中文回答, 条理清晰"""

    prompt = f"请分析: {topic} (市场: {market}){data_context}"

    reply = call_llm(prompt, cfg["api_key"], cfg["base_url"], cfg["model"],
                     system_prompt=system_prompt)
    return reply or f"LLM分析失败, 框架: {skill_name}"


def _action_wind_search(params: dict) -> str:
    question = params.get("question", "")
    market = params.get("market", "A")

    if not question:
        return "⚠️ 请提供筛选条件"

    raw_data = _call_wind_mcp("stock_data", "search_stocks", {"question": question})

    if not raw_data:
        return f"⚠️ Wind MCP 选股筛选失败"

    from src.recommend.advisor import call_llm
    from src.utils.config import get_llm_config
    cfg = get_llm_config()

    prompt = f"""请根据以下Wind MCP选股筛选结果，简洁总结(200字以内):
筛选条件: {question}
数据: {raw_data}

请用中文回答，列出关键股票及核心指标。"""

    reply = call_llm(prompt, cfg["api_key"], cfg["base_url"], cfg["model"],
                     system_prompt="你是投资数据分析师，根据Wind筛选数据简洁回答，不要编造数据。")
    return reply or f"筛选结果: {raw_data[:500]}"


def _action_scan(args: str) -> str:
    from src.recommend.batch_scan import batch_scan, parse_stocks_arg
    from src.data.fetcher import load_watchlist

    if not args.strip():
        items = load_watchlist()
        if not items:
            return "⚠️ 自选列表为空"
        stock_list = [(i.symbol, i.market, i.name) for i in items]
        fast = True
    else:
        stock_list = _resolve_stocks_from_args(args)
        if not stock_list:
            stock_str = args.strip()
            try:
                stock_list = parse_stocks_arg(stock_str)
            except Exception:
                return "格式: /scan 代码:市场:名称,代码:市场:名称\n例: /scan 600519:A:茅台,688519:A"
        fast = len(stock_list) > 5

    try:
        results, summary, reports = batch_scan(
            stocks=stock_list, steps=30, capital=100000,
            import_to_watchlist_flag=False,
            run_all_models=not fast, run_all_strategies=not fast,
            bullish_only_strategies=True,
        )
        if len(summary) > 3000:
            summary = summary[:3000] + "\n...(完整报告见app)"
        return summary
    except Exception as e:
        return f"⚠️ 扫描失败: {e}"


def _parse_stock_args(args: str) -> Optional[tuple]:
    """解析股票参数: '600519:A:贵州茅台' 或 '600519 A 茅台'"""
    args = args.strip()
    m = re.match(r'^(\d{6})\s*[:\s]\s*([A-Z])\s*[:\s]\s*(.+)$', args)
    if m:
        return (m.group(1), m.group(2), m.group(3).strip())
    m = re.match(r'^(\d{6})\s+([A-Z])\s+(.+)$', args)
    if m:
        return (m.group(1), m.group(2), m.group(3).strip())
    m = re.match(r'^(\d{6})\s*[:\s]\s*([A-Z])$', args)
    if m:
        return (m.group(1), m.group(2), m.group(1))
    m = re.match(r'^(\d{6})$', args)
    if m:
        return (m.group(1), "A", m.group(1))
    return None


def _resolve_stocks_from_args(args: str) -> list[tuple[str, str, str]]:
    """从参数解析股票列表，无参数则返回全自选"""
    from src.data.fetcher import load_watchlist

    args = args.strip()
    if not args:
        items = load_watchlist()
        return [(i.symbol, i.market, i.name) for i in items]

    parsed = _parse_stock_args(args)
    if parsed:
        return [parsed]

    stock_list = []
    for segment in args.split(","):
        p = _parse_stock_args(segment.strip())
        if p:
            stock_list.append(p)
    return stock_list


def parse_message(message: str) -> Optional[AgentTask]:
    """
    解析群组消息为 agent 任务

    支持的格式:
    - /add 600519:A:茅台        → 直接添加自选股
    - /remove 600519             → 直接移除自选股
    - /watchlist                 → 查看自选列表
    - /alert add/list/remove     → 直接操作监控
    - /strategy 600519:A        → 直接策略推荐
    - /predict 600519:A         → 直接模型预测
    - /scan 600519:A,688519:A   → 直接批量扫描
    - /pm review                 → PM Agent 全量审查
    - /trader daily              → Trader Agent 每日报告
    - /review                    → PM Agent
    - /daily                     → Trader Agent
    - 自然语言                   → Chat LLM
    """
    msg = message.strip()
    if not msg:
        return None

    # 1. 直接操作命令: /add, /remove, /watchlist, /alert, /strategy, /predict, /scan
    cmd_match = re.match(r'^/(\w+)(?:\s+(.+))?$', msg)
    if cmd_match:
        cmd_key = cmd_match.group(1).lower()
        cmd_args = cmd_match.group(2) or ""

        if cmd_key in DIRECT_ACTION_COMMANDS:
            return AgentTask(
                source="group", group_name="", sender="",
                message=msg, agent="action",
                command=f"{cmd_key} {cmd_args}",
            )

        # Agent commands: /<agent> <command> [args]
        if cmd_key in COMMAND_MAP and cmd_args:
            cmd_info = COMMAND_MAP[cmd_key]
            return AgentTask(
                source="group", group_name="", sender="",
                message=msg, agent=cmd_info["agent"],
                command=cmd_info["prompt"] + " " + cmd_args,
            )

        if cmd_key in AGENT_MAP and cmd_args:
            agent_key = cmd_key
            cmd_lower = cmd_args.split()[0].lower()
            extra_args = " ".join(cmd_args.split()[1:])
            if cmd_lower in COMMAND_MAP:
                prompt = COMMAND_MAP[cmd_lower]["prompt"]
                if extra_args:
                    prompt += f"，重点关注: {extra_args}"
            else:
                prompt = cmd_args
            return AgentTask(
                source="group", group_name="", sender="",
                message=msg, agent=agent_key, command=prompt,
            )

        # Shortcut: /review, /daily, etc.
        if cmd_key in COMMAND_MAP:
            cmd_info = COMMAND_MAP[cmd_key]
            return AgentTask(
                source="group", group_name="", sender="",
                message=msg, agent=cmd_info["agent"],
                command=cmd_info["prompt"],
            )

        if cmd_key in AGENT_MAP:
            return AgentTask(
                source="group", group_name="", sender="",
                message=msg, agent=cmd_key, command="分析自选股",
            )

    # 2. Natural language → 先尝试意图路由(识别为操作则直接执行), 否则走chat
    try:
        intent_task = _route_intent(msg)
        if intent_task:
            logger.info(f"Intent routed natural language to action: {intent_task.command}")
            return intent_task
    except Exception as e:
        logger.warning(f"Intent routing failed, falling back to chat: {e}")

    return AgentTask(
        source="group", group_name="", sender="",
        message=msg, agent="chat", command=msg,
    )


def execute_task(task: AgentTask) -> str:
    logger.info(f"Executing task {task.task_id}: agent={task.agent}, command={task.command[:100]}")

    # chain tasks already have result computed by _route_intent
    if task.agent == "action" and task.command.startswith("chain:") and task.result:
        task.status = "completed"
        task.completed_at = datetime.now().isoformat(timespec="seconds")
        _save_task(task)
        return task.result

    task.status = "running"
    _save_task(task)

    # action_deferred: 延迟执行 (intent JSON 在 command 字段, 后台线程执行)
    if task.agent == "action_deferred":
        try:
            intents = json.loads(task.command)
            results = []
            for item in intents:
                try:
                    intent = item.get("intent", "")
                    params = item.get("params", {})
                    r = _execute_intent_action(intent, params)
                    results.append(r)
                except Exception as e:
                    results.append(f"⚠️ {item.get('intent')} 执行失败: {e}")
            task.result = "\n\n".join(results)
            task.status = "completed"
        except Exception as e:
            task.result = f"意图执行失败: {e}"
            task.status = "failed"
        task.completed_at = datetime.now().isoformat(timespec="seconds")
        _save_task(task)
        return task.result

    if task.agent == "action":
        parts = task.command.split(None, 1)
        cmd = parts[0] if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        result = _execute_direct_action(cmd, args)
        task.result = result
        task.status = "completed"
        task.completed_at = datetime.now().isoformat(timespec="seconds")
        _save_task(task)
        return task.result

    # chat agent 直接用 fallback (LLM回答), 不走 opencode CLI
    if task.agent == "chat":
        fallback_result = _execute_chat_fallback(task.command)
        if fallback_result:
            task.result = fallback_result
            task.status = "completed"
        else:
            task.result = "LLM 调用失败"
            task.status = "failed"
        task.completed_at = datetime.now().isoformat(timespec="seconds")
        _save_task(task)
        return task.result

    # 其他 agent: 尝试 opencode CLI, 失败则 fallback
    agent_file = AGENT_MAP[task.agent]["agent_file"]
    prompt = task.command

    logger.info(f"Executing task {task.task_id}: agent={task.agent}, prompt={prompt[:100]}")

    task.status = "running"
    _save_task(task)

    try:
        # Use opencode CLI to dispatch to the subagent
        cmd = [
            sys.executable, "-m", "opencode",
            "--agent", agent_file,
            "--prompt", prompt,
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=300, cwd=str(Path(__file__).resolve().parent.parent.parent),
        )

        if result.returncode == 0:
            task.result = result.stdout
            task.status = "completed"
        else:
            # opencode CLI 失败 → fallback
            logger.info(f"opencode CLI exit={result.returncode}, using fallback")
            fallback_result = _execute_fallback(task)
            if fallback_result:
                task.result = fallback_result
                task.status = "completed"
            else:
                task.result = result.stderr or f"执行失败 (exit code {result.returncode})"
                task.status = "failed"

    except subprocess.TimeoutExpired:
        task.result = "任务超时 (5分钟限制)"
        task.status = "timeout"
    except Exception as e:
        logger.info(f"opencode CLI 不可用, 使用 fallback: {e}")
        fallback_result = _execute_fallback(task)
        if fallback_result:
            task.result = fallback_result
            task.status = "completed"
        else:
            task.result = f"执行失败: {e}"
            task.status = "failed"

    task.completed_at = datetime.now().isoformat(timespec="seconds")
    _save_task(task)

    logger.info(f"Task {task.task_id} completed: status={task.status}")
    return task.result


def _execute_fallback(task: AgentTask) -> str:
    """
    Fallback: 当 opencode CLI 不可用时，直接调用 agent 的 Python 逻辑
    """
    if task.agent == "chat":
        return _execute_chat_fallback(task.command)
    if task.agent == "trader":
        return _execute_trader_fallback(task.command)
    elif task.agent == "pm":
        return _execute_pm_fallback(task.command)
    return "未知 agent 类型"

CHAT_SYSTEM_PROMPT = """你是 StockPredict 智能投顾助手，精通A股/港股/美股市场分析。

你有以下分析框架可用，根据用户问题自动选择合适的框架:
- Serenity Skill: 供应链瓶颈猎手，从产业链卡点寻找结构性投资机会
- Stock Skill: 三位美股操盘手框架 (Serenity×TraderS×小猫猫)
- Wind MCP: Wind实时数据(行情/基本面/选股筛选)
- 量化模型: GBDT/XGBoost/LSTM预测 + 8个回测策略

回答规则:
1. 直接回答用户的问题，简洁专业
2. 与投资无关的问题，礼貌说明你的专业领域
3. 如果问题涉及实时行情/基本面/选股，提醒用户使用 /wind_query 指令
4. 如果问题涉及产业链分析/瓶颈/Serenity，注入对应框架知识回答"""

INTENT_SYSTEM_PROMPT = """你是 StockPredict 的意图识别引擎。你的任务是判断用户的自然语言消息是否对应可执行的操作。用户可能同时表达多个意图(如"跑策略+设监控")，你需要按顺序列出所有意图。

可用操作列表:
- add: 添加自选股 (参数: 代码, 市场, 名称)
- remove: 移除自选股 (参数: 代码)
- watchlist: 查看自选列表 (无参数)
- alert_add: 添加监控规则 (参数: 代码, 市场, 条件类型, 条件参数, 标签)
- alert_list: 查看所有监控规则 (无参数)
- alert_remove: 移除监控规则 (参数: uid)
- strategy: 策略推荐 (参数: 代码, 市场)
- predict: 模型预测 (参数: 代码, 市场)
- scan: 批量扫描 (参数: 股票列表 或 空=全自选)
- wind_query: 调用Wind MCP获取实时行情/基本面/选股筛选数据 (参数: 代码, 市场, 查询类型)
- wind_search: 调用Wind MCP选股筛选 (参数: 筛选条件, 市场)
- serenity_analysis: 用Serenity供应链瓶颈框架分析产业链/主题 (参数: 主题/股票/行业, 市场)
- stock_analysis: 用三位操盘手框架(Serenity×TraderS×小猫猫)做综合分析 (参数: 股票/主题, 市场)

常用股票代码:
贵州茅台=600519, 航天电器=002025, 南亚新材=688519, 奕东电子=301123, 宁德时代=300750, 比亚迪=002594, 中芯国际=688981,
三环集团=300408, 国瓷材料=300285, 博迁新材=605376, 风华高科=000636, 洁美科技=002859,
寒武纪=688256, 沪电股份=002463, 深南电路=002916, 北方华创=002371, 兆易创新=603986, 立讯精密=002475,
工业富联=601138, 京东方A=000725, 长江电力=600900, 洛阳钼业=603993, 腾讯控股=0700

Wind代码格式:
A股: 600519→600519.SH, 002025→002025.SZ, 300750→300750.SZ, 688519→688519.SH
港股: 0700→00700.HK
美股: AAPL→AAPL.O

可用监控条件类型:
golden_cross(金叉), death_cross(死叉), above_ma(上穿均线), below_ma(下穿均线), rsi_oversold(RSI超卖), rsi_overbought(RSI超买), bollinger_upper(布林上轨), bollinger_lower(布林下轨), daily_change(日涨跌幅), volume_spike(放量), ma_cross_combo(均线组合), rsi_combo(RSI组合), bollinger_combo(布林组合)

市场类型: A(A股), HK(港股), US(美股)

Wind查询类型:
- quote: 实时行情报价(最新价/涨跌幅/成交量等)
- kline: K线日线数据(近期走势)
- basicinfo: 公司基本档案(行业/市值/主营业务等)
- financial: 财务指标(营收/利润/ROE等)
- search: 选股筛选(按条件筛选股票)

判断规则:
- "添加/加入/加到自选/关注" → add
- "删除/移除/取消关注" → remove
- "看看自选/自选列表/我的股票" → watchlist
- "监控/提醒/报警/盯住/设置监控" → alert_add
- "查看监控/看看提醒" → alert_list
- "策略/推荐策略/用什么策略/怎么操作/回测" → strategy
- "预测/走势预测/未来价格" → predict
- "全量分析/全面分析/预测和回测/跑一下分析" → 同时 predict + strategy
- "扫描/批量分析" → scan
- "实时行情/最新价/现在什么价/涨跌幅/报价/今天涨了" → wind_query(quote)
- "近期走势/K线/日线/走势图" → wind_query(kline)
- "公司档案/基本面/主营业务/行业/市值" → wind_query(basicinfo)
- "财务数据/营收/利润/ROE/财报" → wind_query(financial)
- "筛选/选股/帮我找/符合条件的股票" → wind_search
- "用Serenity/供应链/卡点/瓶颈/产业链分析/深度调研" → serenity_analysis
- "用操盘手框架/Serenity×TraderS/小猫猫/综合研判" → stock_analysis
- 纯闲聊/观点/大盘分析/知识问答 → chat

你必须严格按以下JSON格式回复，不要输出任何其他内容:
{"intents": [{"intent": "操作名", "params": {"key": "value"}, "reason": "解释"}, ...]}

如果无法确定参数(如股票代码),从常用代码表查找,找不到则填null。
多个意图按执行顺序排列。纯聊天意图单独标记。

示例:
"把茅台加到自选" → {"intents": [{"intent": "add", "params": {"symbol": "600519", "market": "A", "name": "贵州茅台"}, "reason": "添加茅台到自选"}]}
"帮我盯住688519的金叉" → {"intents": [{"intent": "alert_add", "params": {"symbol": "688519", "market": "A", "condition": "golden_cross", "params": {"short": 5, "long": 20}, "label": "688519金叉"}, "reason": "监控688519金叉"}]}
"航天电器现在什么价" → {"intents": [{"intent": "wind_query", "params": {"symbol": "002025", "market": "A", "query_type": "quote"}, "reason": "查询实时行情"}]}
"帮我找市值超500亿的半导体股票" → {"intents": [{"intent": "wind_search", "params": {"question": "筛选沪深市场市值超500亿的半导体股票", "market": "A"}, "reason": "选股筛选"}]}
"用Serenity框架分析AI半导体产业链" → {"intents": [{"intent": "serenity_analysis", "params": {"topic": "AI半导体产业链", "market": "A"}, "reason": "Serenity供应链分析"}]}
"用操盘手框架分析一下宁德时代" → {"intents": [{"intent": "stock_analysis", "params": {"symbol": "300750", "market": "A", "topic": "宁德时代"}, "reason": "三位操盘手框架分析"}]}
"跑一下三环集团、国瓷材料和博迁新材的全量预测和回测分析" → {"intents": [{"intent": "predict", "params": {"symbol": "300408", "market": "A"}, "reason": "预测"}, {"intent": "strategy", "params": {"symbol": "300408", "market": "A"}, "reason": "回测"}, {"intent": "predict", "params": {"symbol": "300285", "market": "A"}, "reason": "预测"}, {"intent": "strategy", "params": {"symbol": "300285", "market": "A"}, "reason": "回测"}, {"intent": "predict", "params": {"symbol": "605376", "market": "A"}, "reason": "预测"}, {"intent": "strategy", "params": {"symbol": "605376", "market": "A"}, "reason": "回测"}]}
"今天大盘怎么样" → {"intents": [{"intent": "chat", "params": {}, "reason": "纯市场观点讨论"}]}"""


def _route_intent(message: str) -> Optional[AgentTask]:
    from src.recommend.advisor import call_llm
    from src.utils.config import get_llm_config

    cfg = get_llm_config()
    if not cfg["api_key"]:
        return None

    intent_reply = call_llm(message, cfg["api_key"], cfg["base_url"], cfg["model"],
                            system_prompt=INTENT_SYSTEM_PROMPT)
    if not intent_reply:
        return None

    try:
        intent_reply = intent_reply.strip()
        if intent_reply.startswith("```"):
            intent_reply = re.sub(r'^```(?:json)?\s*', '', intent_reply)
            intent_reply = re.sub(r'\s*```$', '', intent_reply)
        intent_data = json.loads(intent_reply)
    except json.JSONDecodeError:
        logger.info(f"Intent parse failed, raw: {intent_reply[:200]}")
        return None

    intents_raw = intent_data.get("intents", [intent_data])
    all_chat = all(i.get("intent") == "chat" for i in intents_raw)
    if all_chat:
        return None

    actionable = [i for i in intents_raw if i.get("intent") != "chat" and i.get("params")]
    if not actionable:
        return None

    # 延迟执行: 只做意图识别, 具体执行放到后台线程 (避免阻塞Flask请求)
    intents_json = json.dumps(actionable, ensure_ascii=False)
    return AgentTask(
        source="group", group_name="", sender="",
        message=message, agent="action_deferred",
        command=intents_json,
    )


def _execute_intent_action(intent: str, params: dict) -> str:
    if intent == "add":
        symbol = params.get("symbol", "")
        market = params.get("market", "A")
        name = params.get("name", symbol)
        if not symbol:
            return "⚠️ 无法识别股票代码"
        return _action_add_stock(f"{symbol}:{market}:{name}")

    elif intent == "remove":
        symbol = params.get("symbol", "")
        if not symbol:
            return "⚠️ 请提供股票代码"
        return _action_remove_stock(symbol)

    elif intent == "watchlist":
        return _action_watchlist("")

    elif intent == "alert_add":
        symbol = params.get("symbol", "")
        market = params.get("market", "A")
        condition = params.get("condition", "golden_cross")
        cond_params = params.get("params", {})
        label = params.get("label", "")
        if not symbol:
            return "⚠️ 请提供股票代码"
        param_str = ",".join(f"{k}={v}" for k, v in cond_params.items()) if cond_params else ""
        return _action_alert(f"add {symbol}:{market}:{condition}:{param_str}:{label}")

    elif intent == "alert_list":
        return _action_alert("list")

    elif intent == "alert_remove":
        uid = params.get("uid", "")
        if not uid:
            return "⚠️ 请提供规则UID，先用/alert list查看"
        return _action_alert(f"remove {uid}")

    elif intent == "alert_toggle":
        uid = params.get("uid", "")
        if not uid:
            return "⚠️ 请提供规则UID"
        return _action_alert(f"toggle {uid}")

    elif intent == "strategy":
        symbol = params.get("symbol", "")
        market = params.get("market", "A")
        if not symbol:
            return _action_strategy("")
        return _action_strategy(f"{symbol}:{market}")

    elif intent == "predict":
        symbol = params.get("symbol", "")
        market = params.get("market", "A")
        if not symbol:
            return _action_predict("")
        return _action_predict(f"{symbol}:{market}")

    elif intent == "scan":
        return _action_scan(params.get("stocks_str", ""))

    elif intent == "wind_query":
        return _action_wind_query(params)

    elif intent == "wind_search":
        return _action_wind_search(params)

    elif intent == "serenity_analysis":
        return _action_skill_analysis(params, "serenity-skill")

    elif intent == "stock_analysis":
        return _action_skill_analysis(params, "stock-skill")

    return f"⚠️ 未知意图: {intent}"


SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "skills"

SKILL_SUMMARIES = {
    "stock-skill": "三位美股操盘手框架: Serenity(供应链卡脖子) × TraderS(宏观判断) × 小猫猫(技术执行)",
    "serenity-skill": "供应链瓶颈猎手: 寻找产业链中不可替代的卡点, 识别结构性投资机会",
    "qlib-skill": "Microsoft Qlib: 因子选股 + 20+ML模型 + Alpha因子挖掘 + 组合优化",
    "finrl-skill": "深度强化学习交易: PPO/A2C/SAC/TD3/DDPG策略训练与回测",
    "investing-algorithms-skill": "量化策略库: 均值回归/动量/配对交易/统计套利等",
    "wind-mcp-skill": "Wind金融数据: 实时行情/基本面/选股筛选/公告新闻",
}


def _build_skill_prompt(command: str) -> str:
    cmd_lower = command.lower()
    relevant = []
    for skill_name, summary in SKILL_SUMMARIES.items():
        keywords = skill_name.replace("-", " ").replace("skill", "").split()
        if any(kw in cmd_lower for kw in keywords):
            relevant.append(f"- {skill_name}: {summary}")
    if any(w in cmd_lower for w in ["供应链", "卡脖子", "瓶颈", "产业链", "不可替代", "卡点", "瓶颈"]):
        relevant.append("- serenity-skill: 供应链瓶颈猎手")
    if any(w in cmd_lower for w in ["因子", "选股", "qlib", "alpha", "组合优化"]):
        relevant.append("- qlib-skill: 因子选股+ML模型+Alpha挖掘")
    if any(w in cmd_lower for w in ["强化学习", "drl", "ppo", "rl", "回测策略"]):
        relevant.append("- finrl-skill: DRL交易策略训练与回测")
    if any(w in cmd_lower for w in ["均值回归", "动量", "配对", "套利", "策略库", "策略对比"]):
        relevant.append("- investing-algorithms-skill: 量化策略库")
    if any(w in cmd_lower for w in ["行情", "基本面", "公告", "新闻", "估值", "筛选"]):
        relevant.append("- wind-mcp-skill: Wind金融数据")
    if any(w in cmd_lower for w in ["操盘", "判断", "框架", "大佬", "serenity", "traders", "恨铁"]):
        relevant.append("- stock-skill: 三位操盘手框架: Serenity × TraderS × 小猫猫")

    skill_context = ""
    if relevant:
        skill_context = "\n\n可参考的投资分析框架:\n" + "\n".join(relevant)
    for s in [x.split(":")[0].replace("- ", "").strip() for x in relevant]:
        skill_file = SKILLS_DIR / f"{s}.md"
        if skill_file.exists():
            content = skill_file.read_text()
            skill_context += f"\n\n--- {s} 分析框架 ---\n{content}"
    return skill_context


def _execute_chat_fallback(command: str) -> str:
    from src.recommend.advisor import call_llm
    from src.utils.config import get_llm_config

    cfg = get_llm_config()
    if not cfg["api_key"]:
        return "⚠️ LLM API 未配置"

    skill_context = _build_skill_prompt(command)
    system_prompt = CHAT_SYSTEM_PROMPT + skill_context

    reply = call_llm(command, cfg["api_key"], cfg["base_url"], cfg["model"], system_prompt=system_prompt)
    if reply:
        return reply
    return "LLM 调用失败，请检查 API 配置"


def _execute_trader_fallback(command: str) -> str:
    """Trader Agent fallback: 基于项目现有模块执行分析"""
    try:
        # 扫描模式: 找到股票代码列表
        is_scan = any(kw in command.lower() for kw in ["扫描", "scan", "批量", "导入", "import"])
        stocks_in_cmd = re.findall(r'(\d{6})', command)

        if is_scan and stocks_in_cmd:
            from src.recommend.batch_scan import batch_scan, parse_stocks_arg
            stock_str = ",".join(f"{code}:A" for code in stocks_in_cmd)
            stock_list = parse_stocks_arg(stock_str)
            results, summary, reports = batch_scan(
                stocks=stock_list, steps=30, capital=100000,
                import_to_watchlist_flag=True,
                run_all_models=False, run_all_strategies=False,
                bullish_only_strategies=True,
            )
            recommended = [r for r in results if r["tag"] in ("强烈推荐", "值得关注")]
            if len(summary) > 4000:
                summary = summary[:4000] + "\n\n... (完整报告见 data/batch_scan/summary.txt)"
            return summary

        # 默认: 自选股快报
        from src.data.fetcher import fetch_data, load_watchlist
        from src.risk.metrics import calc_all_risk_metrics
        from src.utils.config import DATA_DIR

        watchlist_file = DATA_DIR / "watchlist.json"
        if not watchlist_file.exists():
            return "⚠️ 自选股列表为空，请先在app中添加自选股"

        watchlist = json.loads(watchlist_file.read_text())
        if not watchlist:
            return "⚠️ 自选股列表为空"

        results = []
        for item in watchlist[:10]:
            symbol = item.get("symbol", "")
            market = item.get("market", "A")
            name = item.get("name", symbol)
            group = item.get("group", "默认")

            try:
                df = fetch_data(symbol, market, period_days=120, use_cache=True)
                if df.empty or len(df) < 30:
                    continue

                risk = calc_all_risk_metrics(df)
                price = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                change = (price - prev) / prev * 100

                results.append(
                    f"**{name}** ({market}:{symbol}) "
                    f"现价{price:.2f} {change:+.1f}% "
                    f"Sharpe={risk.get('SharpeRatio', 0):.2f} "
                    f"MaxDD={risk.get('MaxDrawdown', 0)*100:.1f}% "
                    f"VaR={risk.get('VaR(95%)', 0)*100:.2f}%"
                )
            except Exception as e:
                results.append(f"⚠️ {symbol} 分析失败: {e}")

        header = f"# 📊 每日自选股快报 — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        return header + "\n".join(results)

    except Exception as e:
        return f"Trader fallback 执行失败: {e}"


def _execute_pm_fallback(command: str) -> str:
    """PM Agent fallback: 快速代码审查"""
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        app_file = project_root / "webapp" / "app.py"

        if not app_file.exists():
            return "⚠️ app.py 不存在"

        code = app_file.read_text()
        lines = code.count("\n") + 1

        # Quick syntax check
        import ast
        try:
            ast.parse(code)
            syntax_ok = True
        except SyntaxError as e:
            syntax_ok = False
            syntax_error = str(e)

        # Check for known issues patterns
        issues = []
        if "st.stop()" in code:
            stop_count = code.count("st.stop()")
            issues.append(f"🟡 发现 {stop_count} 处 st.stop() — 可能阻断其他tab渲染")

        duplicate_blocks = re.findall(r'(with st\.expander\("[^"]+PushPlus[^"]*".*?\n(?:.*?\n)*?st\.rerun\(\))', code)
        if len(duplicate_blocks) > 1:
            issues.append(f"🔴 发现重复 PushPlus UI 块")

        result = f"# 🔍 PM 快速审查 — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        result += f"- app.py: {lines} 行\n"
        result += f"- 语法检查: {'✅ 通过' if syntax_ok else f'❌ {syntax_error}'}\n"
        if issues:
            result += f"\n## 发现的问题:\n" + "\n".join(f"  {i}" for i in issues)
        else:
            result += "\n✅ 未发现明显问题"
        return result

    except Exception as e:
        return f"PM fallback 执行失败: {e}"


def format_reply(task: AgentTask) -> str:
    """将 agent 结果格式化为群组回复消息"""
    agent_name = AGENT_MAP.get(task.agent, {}).get("name", "StockPredict助手")

    if task.status == "completed":
        result = task.result
        if len(result) > 4000:
            result = result[:4000] + "\n\n... (结果过长，已截断，完整结果见app)"
        if task.agent in ("chat", "action", "action_deferred"):
            return result
        return f"## 🤖 {agent_name} 执行结果\n\n**任务**: {task.message}\n\n{result}"
    elif task.status == "timeout":
        return f"⚠️ {agent_name} 任务超时 (5分钟)，请稍后重试或简化指令"
    elif task.status == "failed":
        return f"❌ {agent_name} 执行失败: {task.result[:200]}"
    else:
        return f"⏳ {agent_name} 正在执行中..."


def _save_task(task: AgentTask):
    """持久化任务记录"""
    TASK_LOG.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    if TASK_LOG.exists():
        try:
            tasks = json.loads(TASK_LOG.read_text())
        except Exception:
            pass
    tasks.append({
        "task_id": task.task_id,
        "source": task.source,
        "message": task.message,
        "agent": task.agent,
        "command": task.command,
        "status": task.status,
        "result": task.result[:500] if task.result else "",
        "created_at": task.created_at,
        "completed_at": task.completed_at,
    })
    # Keep last 200 tasks
    tasks = tasks[-200:]
    TASK_LOG.write_text(json.dumps(tasks, ensure_ascii=False, indent=2))


def load_recent_tasks(limit: int = 20) -> list[dict]:
    """加载最近的任务记录"""
    if not TASK_LOG.exists():
        return []
    try:
        tasks = json.loads(TASK_LOG.read_text())
        return tasks[-limit:]
    except Exception:
        return []
