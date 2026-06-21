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


def parse_message(message: str) -> Optional[AgentTask]:
    """
    解析群组消息为 agent 任务
    
    支持的格式:
    - /pm review              → PM Agent 全量审查
    - /trader daily           → Trader Agent 每日报告
    - /review                 → 默认 PM Agent
    - /daily                  → 默认 Trader Agent
    - /trader 分析 603186    → Trader Agent 分析指定股票
    - /pm 测试回测模块       → PM Agent 测试指定模块
    - 分析自选股              → 自动识别为 Trader 任务
    - 检查app bug             → 自动识别为 PM 任务
    """
    msg = message.strip()
    if not msg:
        return None

    # 1. 命令格式: /<agent> <command> [args]
    cmd_match = re.match(r'^/(\w+)\s+(.+)$', msg)
    if cmd_match:
        agent_key = cmd_match.group(1).lower()
        command_text = cmd_match.group(2).strip()

        # /<command> without explicit agent (e.g. /review, /daily)
        if agent_key in COMMAND_MAP:
            cmd_info = COMMAND_MAP[agent_key]
            return AgentTask(
                source="group", group_name="", sender="",
                message=msg, agent=cmd_info["agent"],
                command=cmd_info["prompt"] + " " + command_text,
            )

        # /<agent_key> <command>
        if agent_key in AGENT_MAP:
            # Check if command_text matches a known command
            cmd_lower = command_text.split()[0].lower()
            extra_args = " ".join(command_text.split()[1:]) if len(command_text.split()) > 1 else ""

            if cmd_lower in COMMAND_MAP:
                prompt = COMMAND_MAP[cmd_lower]["prompt"]
                if extra_args:
                    prompt += f"，重点关注: {extra_args}"
            else:
                prompt = command_text

            return AgentTask(
                source="group", group_name="", sender="",
                message=msg, agent=agent_key, command=prompt,
            )

    # 2. 命令格式: /<command> (shortcut)
    shortcut_match = re.match(r'^/(\w+)$', msg)
    if shortcut_match:
        cmd = shortcut_match.group(1).lower()
        if cmd in COMMAND_MAP:
            cmd_info = COMMAND_MAP[cmd]
            return AgentTask(
                source="group", group_name="", sender="",
                message=msg, agent=cmd_info["agent"],
                command=cmd_info["prompt"],
            )

    # 3. Natural language: keyword matching
    msg_lower = msg.lower()
    best_agent = None
    best_score = 0

    for agent_key, agent_info in AGENT_MAP.items():
        score = sum(1 for kw in agent_info["keywords"] if kw in msg_lower)
        if score > best_score:
            best_score = score
            best_agent = agent_key

    if best_agent and best_score >= 1:
        if best_agent == "pm":
            prompt = f"用户请求: {msg}。请系统性审查app相关部分，输出结构化报告"
        else:
            prompt = f"用户请求: {msg}。请执行完整的交易分析流程"
        return AgentTask(
            source="group", group_name="", sender="",
            message=msg, agent=best_agent, command=prompt,
        )

    return None


def execute_task(task: AgentTask) -> str:
    """
    执行 agent 任务 — 通过 opencode CLI 调用 subagent
    
    使用 opencode 的 --agent 和 --prompt 参数
    """
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
    
    这里模拟 agent 的核心分析能力
    """
    if task.agent == "trader":
        return _execute_trader_fallback(task.command)
    elif task.agent == "pm":
        return _execute_pm_fallback(task.command)
    return "未知 agent 类型"


def _execute_trader_fallback(command: str) -> str:
    """Trader Agent fallback: 基于项目现有模块执行分析"""
    try:
        from src.data.fetcher import fetch_data
        from src.data.stock_db import get_db
        from src.models.factory import run_models
        from src.risk.metrics import calc_all_risk_metrics
        from src.utils.config import load_config, DATA_DIR

        db = get_db()
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
    agent_name = AGENT_MAP[task.agent]["name"]

    if task.status == "completed":
        result = task.result
        if len(result) > 4000:
            result = result[:4000] + "\n\n... (结果过长，已截断，完整结果见app)"
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
