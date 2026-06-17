import json
from datetime import datetime

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

from src.utils.config import WATCHLIST_FILE, StockItem, DATA_DIR, load_config, save_config, get_tushare_token
from src.data.stock_db import search_stocks, get_stock_name, parse_stock_file
from src.alerts import AlertRule, add_rule, remove_rule, toggle_rule, load_rules, get_engine, CONDITION_TYPES
from src.alerts.settings import MonitorSettings, load_settings, save_settings, MARKET_NAMES
from src.data.fetcher import load_watchlist, save_watchlist, fetch_data, get_realtime_price
from src.models.factory import list_models, run_models, MODEL_REGISTRY
from src.models.gbdt import GBDTModel
from src.risk.metrics import calc_all_risk_metrics
from src.backtesting.engine import BacktestEngine
from src.backtesting.models import BacktestConfig
from src.backtesting.strategies import (
    MovingAverageCrossStrategy, PredictionStrategy, RollingPredictionStrategy,
)

console = Console()


@click.group()
def cli():
    """自选股票股价走势预测与风控工具"""


@cli.command()
@click.argument("symbol", required=False, default=None)
@click.argument("market", type=click.Choice(["A", "HK", "US"]), required=False, default=None)
@click.option("--name", "-n", default="", help="股票名称")
@click.option("--search", "-s", is_flag=True, help="模糊搜索模式")
def add(symbol, market, name, search):
    """添加自选股

    支持多种方式:
      stock add 600519 A       直接添加
      stock add -s 茅台        模糊搜索后选择
      stock add                交互式输入
    """
    from src.data.stock_db import search_stocks, get_stock_name

    items = load_watchlist()

    # ── 搜索模式 ──
    if search or not symbol:
        query = symbol or click.prompt("搜索股票 (代码/名称)", default="")
        if not query:
            console.print("[red]请输入搜索关键词[/red]")
            return
        results = search_stocks(query)

        if not results:
            console.print(f"[yellow]未找到匹配: {query}[/yellow]")
            console.print("试试直接指定: stock add 600519 A")
            return

        # 让用户选择
        console.print(f"找到 {len(results)} 条结果:")
        for i, (code, name, m) in enumerate(results[:15], 1):
            console.print(f"  {i}. [{m}] {code} {name}")

        choice = click.prompt("选择序号 (或直接输入代码)", default="1")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                symbol, name, market = results[idx]
        except ValueError:
            # 直接输入代码
            for code, n, m in results:
                if code == choice:
                    symbol, name, market = code, n, m
                    break
            else:
                console.print("[red]无效选择[/red]")
                return
    else:
        if not name:
            name = get_stock_name(symbol, market)

    if any(i.symbol == symbol and i.market == market for i in items):
        console.print(f"[yellow]已在自选列表中: {market}:{symbol}[/yellow]")
        return

    items.append(StockItem(symbol=symbol, market=market, name=name))
    save_watchlist(items)
    console.print(f"[green]已添加 {market}:{symbol} {name}[/green]")


@cli.command()
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--market", "-m", default=None,
              help="强制指定市场 (A/HK/US), 不指定则自动识别")
def import_stocks(filepath, market):
    """从 CSV/TXT/XLSX 文件导入自选股"""
    from src.data.stock_db import parse_stock_file

    try:
        codes = parse_stock_file(filepath)
    except Exception as e:
        console.print(f"[red]解析失败: {e}[/red]")
        return

    if not codes:
        console.print("[yellow]未解析到任何股票代码[/yellow]")
        return

    items = load_watchlist()
    added = 0
    skipped = 0

    for code, name, m in codes:
        m = market or m
        if not m:
            m = "A"
        if any(i.symbol == code and i.market == m for i in items):
            skipped += 1
            continue
        items.append(StockItem(symbol=code, market=m, name=name))
        added += 1

    save_watchlist(items)
    console.print(f"[green]导入完成: 新增 {added}, 跳过 {skipped}[/green]")
    console.print(f"当前自选共 {len(items)} 只")


@cli.command()
@click.argument("symbol")
@click.argument("market", type=click.Choice(["A", "HK", "US"]))
def remove(symbol: str, market: str):
    """移除自选股"""
    items = load_watchlist()
    items = [i for i in items if not (i.symbol == symbol and i.market == market)]
    save_watchlist(items)
    console.print(f"[green]已移除 {market}:{symbol}[/green]")


@cli.command(name="list")
def list_stocks():
    """列出所有自选股"""
    items = load_watchlist()
    if not items:
        console.print("[yellow]自选列表为空[/yellow]")
        return
    table = Table(box=box.ROUNDED)
    table.add_column("#", style="dim")
    table.add_column("市场")
    table.add_column("代码")
    table.add_column("名称")
    table.add_column("现价", justify="right")
    for i, item in enumerate(items, 1):
        try:
            price = get_realtime_price(item.symbol, item.market)
            price_str = f"{price:.2f}" if price else "N/A"
        except Exception:
            price_str = "N/A"
        table.add_row(str(i), item.market, item.symbol, item.name or "-", price_str)
    console.print(table)


@cli.command()
@click.option("--symbol", "-s", default=None, help="指定股票代码")
@click.option("--market", "-m", default=None, help="指定市场")
@click.option("--models", "-M", default="all", help="预测模型, 逗号分隔")
@click.option("--steps", "-S", default=30, help="预测天数")
@click.option("--plot", "-p", is_flag=True, help="生成预测图")
def predict(symbol, market, models, steps, plot):
    """预测股价走势"""
    items = load_watchlist()
    if symbol and market:
        items = [StockItem(symbol=symbol, market=market)]
    elif not items:
        console.print("[red]自选列表为空，请先用 add 添加[/red]")
        return

    model_names = list_models() if models == "all" else [m.strip() for m in models.split(",")]

    for item in items:
        console.print(f"\n[bold blue]▶ {item.market}:{item.symbol} {item.name}[/bold blue]")
        try:
            df = fetch_data(item.symbol, item.market)
        except Exception as e:
            console.print(f"[red]数据获取失败: {e}[/red]")
            continue

        console.print(f"  数据条数: {len(df)}, 区间: {df['Date'].iloc[0].date()} ~ {df['Date'].iloc[-1].date()}")

        # 风控指标
        risk = calc_all_risk_metrics(df)
        table_risk = Table(title="风控指标", box=box.SIMPLE)
        table_risk.add_column("指标")
        table_risk.add_column("值", justify="right")
        for k, v in risk.items():
            table_risk.add_row(k, str(v))
        console.print(table_risk)

        # 预测
        results = run_models(df, model_names=model_names, steps=steps)

        table_pred = Table(title="预测结果对比", box=box.ROUNDED)
        table_pred.add_column("模型")
        table_pred.add_column("MAE", justify="right")
        table_pred.add_column("RMSE", justify="right")
        table_pred.add_column("MAPE(%)", justify="right")
        table_pred.add_column(f"未来{steps}天方向", justify="center")

        for name, result in results.items():
            m = result.metrics
            if "error" in m:
                table_pred.add_row(name, "[red]失败[/red]", m["error"], "", "")
                continue
            direction = "📈 上涨" if result.forecast[-1] > result.history[-1] else "📉 下跌"
            table_pred.add_row(
                name,
                str(m.get("MAE", "-")),
                str(m.get("RMSE", "-")),
                str(m.get("MAPE", "-")),
                direction,
            )
        console.print(table_pred)

        if plot and results:
            try:
                _plot_results(item, df, results)
            except Exception as e:
                console.print(f"[red]绘图失败: {e}[/red]")


@cli.command()
def diagnose():
    """诊断数据源连接状态"""
    from src.data.fetcher import diagnose_sources

    table = Table(title="📡 数据源诊断", box=box.ROUNDED)
    table.add_column("市场")
    table.add_column("数据源")
    table.add_column("状态")
    table.add_column("延迟")
    table.add_column("说明")

    for market, sources in diagnose_sources().items():
        for i, s in enumerate(sources):
            status = "🟢 可用" if s["available"] else "🔴 不可用"
            latency = f"{s['latency_ms']:.0f}ms" if s["available"] else "-"
            error = s.get("error", "")[:45]
            m_label = market if i == 0 else ""
            table.add_row(m_label, s["name"], status, latency, error)

    console.print(table)
    cache_dir = DATA_DIR / "cache"
    if cache_dir.exists():
        files = list(cache_dir.glob("*.parquet"))
        console.print(f"\n本地缓存: {len(files)} 个文件 ({sum(f.stat().st_size for f in files)//1024} KB)")
    console.print("\n提示: 如果所有真实数据源都不可用, 系统自动使用模拟数据")


def _plot_results(item: StockItem, df, results: dict):
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"{item.market}:{item.symbol} {item.name}")

    # 价格 & 预测
    ax1 = axes[0]
    dates = pd.to_datetime(df["Date"])
    ax1.plot(dates, df["Close"], label="历史收盘价", color="black", linewidth=1)
    colors = {"arima": "red", "lstm": "blue", "gbdt": "green"}
    for name, result in results.items():
        if len(result.forecast) == 0:
            continue
        fdates = result.forecast_dates
        ax1.plot(fdates, result.forecast, label=f"{name}预测",
                 color=colors.get(name, "purple"), linestyle="--", marker=".")
    ax1.set_ylabel("价格")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 成交量
    ax2 = axes[1]
    ax2.bar(dates, df["Volume"] / 1e8, width=1, color="gray", alpha=0.5)
    ax2.set_ylabel("成交量(亿)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"预测图_{item.symbol}_{timestamp}.png"
    plt.savefig(path, dpi=150)
    console.print(f"[green]预测图已保存: {path}[/green]")
    plt.close()


@cli.command()
def models():
    """列出所有可用预测模型"""
    from src.models.factory import _init_registry, MODEL_REGISTRY as M
    _init_registry()
    table = Table(box=box.ROUNDED)
    table.add_column("模型名")
    table.add_column("说明")
    for name in list_models():
        table.add_row(name, M[name].__doc__ or "-")
    console.print(table)


# ─── 配置管理 ─────────────────────────────────────────────
@cli.group()
def config():
    """查看/修改配置"""

@config.command(name="show")
def config_show():
    """显示当前配置"""
    cfg = load_config()
    table = Table(box=box.ROUNDED)
    table.add_column("配置项")
    table.add_column("值")
    table.add_row("Tushare Token", cfg.get("tushare_token", "未设置")[:8] + "****" if cfg.get("tushare_token") else "未设置")
    table.add_row("自选股数量", str(len(load_watchlist())))
    table.add_row("缓存目录", str(DATA_DIR / "cache"))
    console.print(table)


@config.command(name="set-tushare-token")
@click.argument("token")
def config_set_token(token: str):
    """设置 Tushare Token (存储到本地配置文件)"""
    cfg = load_config()
    cfg["tushare_token"] = token
    save_config(cfg)
    console.print(f"[green]Tushare Token 已保存[/green]")
    console.print("也可以通过环境变量设置: export TUSHARE_TOKEN=你的token")


@config.command(name="set-llm-key")
@click.argument("api_key")
def config_set_llm(api_key: str):
    """设置 DeepSeek API Key"""
    cfg = load_config()
    cfg["llm_api_key"] = api_key
    save_config(cfg)
    console.print(f"[green]LLM API Key 已保存[/green]")
    console.print("也可以通过环境变量设置: export DEEPSEEK_API_KEY=你的key")


@config.command(name="set-serverchan-key")
@click.argument("sendkey")
def config_set_sck(sendkey: str):
    """设置 Server酱 SendKey (微信推送)"""
    cfg = load_config()
    cfg["serverchan_key"] = sendkey
    save_config(cfg)
    console.print(f"[green]Server酱 SendKey 已保存[/green]")
    console.print("获取 SendKey: https://sct.ftqq.com")


@cli.command()
@click.argument("symbol")
@click.argument("market", type=click.Choice(["A", "HK", "US"]))
@click.option("--strategy", "-s", default="ma_cross",
              help="策略: ma_cross / prediction / rolling_prediction")
@click.option("--capital", "-c", default=100000, help="初始资金")
@click.option("--warmup", "-w", default=120, help="滚动预测: 初始训练天数")
@click.option("--retrain-freq", "-r", default=20, help="滚动预测: 重训频率(天)")
@click.option("--plot", "-p", is_flag=True, help="生成回测图")
def backtest(symbol, market, strategy, capital, warmup, retrain_freq, plot):
    """回测策略 on 指定股票"""
    try:
        df = fetch_data(symbol, market)
    except Exception as e:
        console.print(f"[red]数据获取失败: {e}[/red]")
        return

    cfg = BacktestConfig(initial_capital=capital, market=market)

    if strategy == "ma_cross":
        strat = MovingAverageCrossStrategy(short=20, long=60)
    elif strategy == "prediction":
        model = GBDTModel(lookback=30)
        strat = PredictionStrategy(model, threshold_buy=0.015, threshold_sell=-0.015)
    elif strategy == "rolling_prediction":
        model = GBDTModel(lookback=30)
        strat = RollingPredictionStrategy(
            model, warmup=warmup, retrain_freq=retrain_freq,
            threshold_buy=0.015, threshold_sell=-0.015,
        )
    else:
        console.print(f"[red]未知策略: {strategy}[/red]")
        return

    engine = BacktestEngine(df, strat, cfg)
    result = engine.run()

    # 绩效表
    table = Table(title=f"回测结果: {market}:{symbol} 策略={strat.name}",
                  box=box.ROUNDED)
    table.add_column("指标")
    table.add_column("值", justify="right")
    for k, v in result.metrics.items():
        table.add_row(k, str(v))
    console.print(table)

    # 交易表
    tdf = result.trades_df()
    if not tdf.empty:
        table2 = Table(title=f"交易记录 (共{len(tdf)}笔)", box=box.SIMPLE)
        for col in tdf.columns:
            table2.add_column(col, justify="right" if col in ("盈亏","收益率") else "left")
        for _, row in tdf.iterrows():
            table2.add_row(*[str(v) for v in row.values])
        console.print(table2)

    # 绘图
    if plot:
        try:
            from src.backtesting.viz import plot_backtest
            path = f"回测图_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_backtest(result, save_path=path, title=f"{market}:{symbol} {strat.name}")
        except Exception as e:
            console.print(f"[red]绘图失败: {e}[/red]")


# ═══════════════════════════════════════════════════════════
#  交易提醒
# ═══════════════════════════════════════════════════════════
@cli.group()
def alert():
    """交易提醒管理"""

@alert.command("add")
@click.argument("symbol")
@click.argument("market", type=click.Choice(["A", "HK", "US"]))
@click.argument("condition", type=click.Choice(list(CONDITION_TYPES.keys())))
@click.option("--params", "-p", default="", help="参数, 用逗号分隔: window=20,threshold=100")
@click.option("--label", "-l", default="", help="备注")
@click.option("--interval", "-i", default=5, help="检查间隔(分钟)")
@click.option("--cooldown", "-c", default=60, help="冷却时间(分钟)")
def alert_add(symbol, market, condition, params, label, interval, cooldown):
    """添加交易提醒规则"""
    from src.alerts import AlertRule, add_rule
    from src.data.stock_db import get_stock_name

    param_dict = {}
    if params:
        for part in params.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    v = float(v)
                    v = int(v) if v == int(v) else v
                except ValueError:
                    pass
                param_dict[k.strip()] = v

    rule = AlertRule(
        symbol=symbol, market=market, condition=condition,
        params=param_dict, label=label or get_stock_name(symbol, market),
        interval_minutes=interval, cooldown_minutes=cooldown,
    )
    add_rule(rule)
    console.print(f"[green]已添加提醒: {rule.summary}[/green]")


@alert.command("list")
def alert_list():
    """列出所有提醒规则"""
    from src.alerts import load_rules, CONDITION_TYPES

    rules = load_rules()
    if not rules:
        console.print("[yellow]暂无提醒规则[/yellow]")
        return

    table = Table(box=box.ROUNDED)
    table.add_column("状态")
    table.add_column("股票")
    table.add_column("条件")
    table.add_column("参数")
    table.add_column("备注")
    table.add_column("冷却剩余")
    table.add_column("UID")

    for r in rules:
        status = "🟢" if r.enabled else "🔴"
        desc = CONDITION_TYPES.get(r.condition, r.condition)
        params = ", ".join(f"{k}={v}" for k, v in r.params.items())
        from datetime import datetime, timedelta
        cooldown_str = ""
        if r.last_triggered:
            last = datetime.fromisoformat(r.last_triggered)
            remaining = timedelta(minutes=r.cooldown_minutes) - (datetime.now() - last)
            if remaining.total_seconds() > 0:
                cooldown_str = f"{int(remaining.total_seconds()//60)}min"
        table.add_row(status, f"[{r.market}] {r.symbol}", desc,
                      params, r.label[:15], cooldown_str, r.uid[:20])
    console.print(table)


@alert.command("remove")
@click.argument("uid")
def alert_remove(uid):
    """删除提醒规则 (使用 UID)"""
    from src.alerts import remove_rule
    if remove_rule(uid):
        console.print(f"[green]已删除规则[/green]")
    else:
        console.print("[red]未找到规则[/red]")


@alert.command("toggle")
@click.argument("uid")
def alert_toggle(uid):
    """启用/禁用规则"""
    from src.alerts import toggle_rule
    r = toggle_rule(uid)
    if r:
        console.print(f"[green]规则已{'启用' if r.enabled else '禁用'}[/green]")
    else:
        console.print("[red]未找到规则[/red]")


@alert.command("start")
def alert_start():
    """启动后台监控"""
    from src.alerts import get_engine
    engine = get_engine()
    engine.start()
    console.print("[green]监控引擎已启动 (30秒间隔)[/green]")


@alert.command("stop")
def alert_stop():
    """停止后台监控"""
    from src.alerts import get_engine
    engine = get_engine()
    engine.stop()
    console.print("[yellow]监控引擎已停止[/yellow]")


@alert.command("status")
def alert_status():
    """查看监控状态"""
    from src.alerts import get_engine, load_rules
    from src.alerts.settings import load_settings
    engine = get_engine()
    rules = load_rules()
    settings = load_settings()
    console.print(f"监控状态: {'🟢 运行中' if engine.is_running else '🔴 已停止'}")
    console.print(f"监控时段: {settings.summary}")
    console.print(f"规则总数: {len(rules)}")
    console.print(f"启用规则: {sum(1 for r in rules if r.enabled)}")
    from pathlib import Path
    log = Path(__file__).resolve().parent.parent.parent / "data" / "alerts.log"
    if log.exists():
        console.print(f"日志文件: {log} ({log.stat().st_size} bytes)")


@alert.command("settings")
@click.option("--market", "-m", type=click.Choice(["A", "HK", "US"]),
              help="参考市场时段")
@click.option("--interval", "-i", type=int, help="检查间隔(分钟)")
@click.option("--start", "-s", help="自定义开始时间 HH:MM")
@click.option("--end", "-e", help="自定义结束时间 HH:MM")
@click.option("--trade-only/--no-trade-only", default=None, help="仅交易日")
@click.option("--show", is_flag=True, help="显示当前设置")
def alert_settings(market, interval, start, end, trade_only, show):
    """查看/修改监控设置"""
    settings = load_settings()
    if show:
        console.print(f"当前监控设置:")
        console.print(f"  参考市场: {MARKET_NAMES.get(settings.market, settings.market)}")
        console.print(f"  交易时段: {settings.summary}")
        console.print(f"  检查间隔: 每{settings.interval_minutes}分钟")
        console.print(f"  仅交易日: {'是' if settings.trade_days_only else '否'}")
        return

    changed = []
    if market:
        settings.market = market
        settings.custom_start = None
        settings.custom_end = None
        changed.append(f"市场={MARKET_NAMES.get(market, market)}")
    if interval:
        settings.interval_minutes = interval
        changed.append(f"间隔={interval}分钟")
    if start:
        settings.custom_start = start
        changed.append(f"开始={start}")
    if end:
        settings.custom_end = end
        changed.append(f"结束={end}")
    if trade_only is not None:
        settings.trade_days_only = trade_only
        changed.append(f"仅交易日={'是' if trade_only else '否'}")

    if changed:
        save_settings(settings)
        console.print(f"[green]设置已更新: {', '.join(changed)}[/green]")
    else:
        console.print("[yellow]使用 --show 查看当前设置[/yellow]")


if __name__ == "__main__":
    cli()
