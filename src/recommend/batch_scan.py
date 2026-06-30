"""
批量扫描模块: 接受任意股票列表 → 导入自选 → 全量预测 → 策略推荐 → 评分排名

对外 API:
  batch_scan(stocks, steps, capital, import_to_watchlist, run_all_models)
    -> (results_json, summary_text, reports_text)

CLI:
  stock batch-scan --stocks "301123:A,688519:A,..."
  stock batch-scan --file stocks.txt
  stock batch-scan --watchlist   (对已有自选股跑扫描)
"""
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fetcher import fetch_data, load_watchlist, save_watchlist
from src.data.stock_db import get_stock_name
from src.utils.config import StockItem
from src.risk.metrics import calc_all_risk_metrics
from src.recommend.engine import ModelResult, generate_report
from src.models.factory import run_models, list_models
from src.backtesting.engine import BacktestEngine
from src.backtesting.models import BacktestConfig
from src.backtesting.strategies import (
    MovingAverageCrossStrategy, RSIStrategy,
    ChannelBreakoutStrategy, BollingerStrategy,
    RollingPredictionStrategy,
)
from src.models.gbdt import GBDTModel

OUTPUT_DIR = Path("data/batch_scan")


ALL_STRATS = [
    ("双均线(5/20)", MovingAverageCrossStrategy(5, 20)),
    ("双均线(10/30)", MovingAverageCrossStrategy(10, 30)),
    ("双均线(20/60)", MovingAverageCrossStrategy(20, 60)),
    ("RSI(14)", RSIStrategy(14, 30, 70)),
    ("通道突破(20/10)", ChannelBreakoutStrategy(20, 10)),
    ("布林带(20/2)", BollingerStrategy(20, 2)),
    ("滚动预测(月频)", RollingPredictionStrategy(
        GBDTModel(), warmup=200, retrain_freq=20,
        threshold_buy=0.015, threshold_sell=-0.015)),
    ("滚动预测(周频)", RollingPredictionStrategy(
        GBDTModel(), warmup=200, retrain_freq=5,
        threshold_buy=0.01, threshold_sell=-0.01)),
]

LIGHT_STRATS = [
    ("双均线(5/20)", MovingAverageCrossStrategy(5, 20)),
    ("双均线(10/30)", MovingAverageCrossStrategy(10, 30)),
    ("双均线(20/60)", MovingAverageCrossStrategy(20, 60)),
    ("RSI(14)", RSIStrategy(14, 30, 70)),
    ("通道突破(20/10)", ChannelBreakoutStrategy(20, 10)),
    ("布林带(20/2)", BollingerStrategy(20, 2)),
]


def parse_stocks_arg(stocks_str: str) -> list[tuple[str, str, str]]:
    """解析股票列表字符串: '301123:A:奕东电子,688519:A' 或 '301123,688519'"""
    result = []
    for part in stocks_str.split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        code = fields[0].strip()
        market = fields[1].strip() if len(fields) > 1 else "A"
        name = fields[2].strip() if len(fields) > 2 else get_stock_name(code, market)
        if code:
            result.append((code, market, name))
    return result


def parse_stocks_file(filepath: str) -> list[tuple[str, str, str]]:
    """从文本文件读取股票列表, 每行格式: 代码:市场:名称 或 代码"""
    result = []
    lines = Path(filepath).read_text().strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        result.extend(parse_stocks_arg(line))
    return result


def import_to_watchlist(stocks: list[tuple[str, str, str]]) -> int:
    """将股票导入自选列表, 返回新增数量"""
    items = load_watchlist()
    added = 0
    for code, market, name in stocks:
        if not any(i.symbol == code and i.market == market for i in items):
            items.append(StockItem(symbol=code, market=market, name=name))
            added += 1
    if added > 0:
        save_watchlist(items)
    return added


def batch_scan(
    stocks: list[tuple[str, str, str]],
    steps: int = 30,
    capital: float = 100000,
    import_to_watchlist_flag: bool = True,
    run_all_models: bool = True,
    run_all_strategies: bool = True,
    bullish_only_strategies: bool = True,
) -> tuple[list[dict], str, str]:
    """
    执行批量扫描:
      1. 导入自选 (可选)
      2. 全量预测
      3. 策略回测 (看涨股跑全量策略, 看跌股跑轻量策略)
      4. 综合评分排名

    Returns: (results_json, summary_text, all_reports_text)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 0: 导入自选
    added = 0
    if import_to_watchlist_flag:
        added = import_to_watchlist(stocks)

    total = len(stocks)
    all_results = []
    all_reports = []
    failed = []

    model_names = list_models() if run_all_models else ["arima", "gbdt", "xgboost"]
    strats = ALL_STRATS if run_all_strategies else LIGHT_STRATS

    print(f"=== 批量扫描 {total} 只股票 ===")
    if added > 0:
        print(f"自选导入: 新增 {added} 只 (已有 {total - added} 只在列表)")
    print(f"模型: {','.join(model_names)} | 策略: {len(strats)}种 | 预测{steps}天")

    # Phase 1: 预测
    print(f"\n--- Phase 1: 预测 ({len(model_names)}个模型) ---")
    pred_data = {}

    for i, (code, market, name) in enumerate(stocks, 1):
        print(f"[{i}/{total}] {market}:{code} {name} ...", end="", flush=True)

        try:
            df = fetch_data(code, market)
        except Exception as e:
            print(f" ❌ 数据失败")
            failed.append((code, name, market, str(e)[:60]))
            continue

        if len(df) < 60:
            print(f" ❌ 数据不足({len(df)})")
            failed.append((code, name, market, f"数据不足{len(df)}条"))
            continue

        current_price = float(df["Close"].iloc[-1])
        print(f" {len(df)}条 ¥{current_price:.2f}", end="", flush=True)

        risk = {}
        try:
            risk = calc_all_risk_metrics(df)
        except Exception:
            pass

        model_results = []
        avg_pct = 0
        consensus = "N/A"
        try:
            raw = run_models(df, model_names=model_names, steps=steps)
            for rname, r in raw.items():
                mr = ModelResult(model=rname)
                if "error" in r.metrics:
                    mr.error = str(r.metrics["error"])[:60]
                elif len(r.forecast) > 0:
                    mr.forecast = r.forecast
                    mr.final_price = float(r.forecast[-1])
                    mr.pct_change = (r.forecast[-1] - r.history[-1]) / r.history[-1] * 100
                    mr.direction = "看涨" if mr.pct_change > 0 else "看跌"
                    mr.mape = float(r.metrics.get("MAPE", 0))
                    mr.data_source = r.data_source
                model_results.append(mr)

            valid_m = [m for m in model_results if not m.error]
            if valid_m:
                avg_pct = np.mean([m.pct_change for m in valid_m])
                up_count = sum(1 for m in valid_m if m.pct_change > 0)
                consensus = f"{up_count}/{len(valid_m)}"
                direction = "看涨" if avg_pct > 0 else "看跌"
                print(f" → {direction} {avg_pct:+.1f}% 共识{consensus}")
            else:
                print(f" → 预测全失败")
        except Exception as e:
            print(f" ❌ {str(e)[:40]}")

        pred_data[code] = {
            "name": name, "market": market,
            "df": df, "current_price": current_price,
            "models": model_results, "risk": risk,
            "avg_pct": avg_pct, "consensus": consensus,
        }

    # Phase 2: 策略回测
    bullish = {k: v for k, v in pred_data.items() if v["avg_pct"] > 0}
    bearish = {k: v for k, v in pred_data.items() if v["avg_pct"] <= 0}

    print(f"\n--- Phase 1 完成: 看涨{len(bullish)} 看跌{len(bearish)} 失败{len(failed)} ---")

    # 看涨股跑全量策略, 看跌股跑轻量策略(省时间)
    scan_targets = []
    if bullish_only_strategies:
        print(f"\n--- Phase 2: 看涨股策略回测 ({len(bullish)}只 × {len(strats)}策略) ---")
        for code, data in bullish.items():
            scan_targets.append((code, data, strats))
        for code, data in bearish.items():
            scan_targets.append((code, data, LIGHT_STRATS))
    else:
        print(f"\n--- Phase 2: 全量策略回测 ({len(pred_data)}只 × {len(strats)}策略) ---")
        for code, data in pred_data.items():
            scan_targets.append((code, data, strats))

    for i, (code, data, strat_list) in enumerate(scan_targets, 1):
        name = data["name"]
        market = data["market"]
        df = data["df"]
        is_bullish = data["avg_pct"] > 0
        strat_set = "全量" if is_bullish else "轻量"
        print(f"[{i}/{len(scan_targets)}] {code} {name} ({strat_set}策略)...", end="", flush=True)

        cfg = BacktestConfig(initial_capital=capital, market=market)
        strat_results = []

        for sname, strat in strat_list:
            try:
                engine = BacktestEngine(df, strat, cfg)
                result = engine.run()
                strat_results.append({
                    "name": sname,
                    "total_return": result.total_return,
                    "sharpe": result.sharpe_ratio,
                    "max_dd": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "trades": result.total_trades,
                })
            except Exception:
                strat_results.append({"name": sname, "error": True})

        best_strat = max(
            [s for s in strat_results if "error" not in s],
            key=lambda s: s["total_return"],
            default=None,
        )
        if best_strat:
            print(f" → 最佳:{best_strat['name']} 收益{best_strat['total_return']*100:+.1f}% 夏普{best_strat['sharpe']:.2f}")
        else:
            print(f" → 策略全失败")

        data["strategies"] = strat_results
        data["best_strategy"] = best_strat

    # Phase 3: 评分
    print(f"\n--- Phase 3: 综合评分 ---")

    for code, data in pred_data.items():
        models = data["models"]
        avg_pct = data["avg_pct"]
        best_s = data.get("best_strategy")

        valid_m = [m for m in models if not m.error]
        up_count = sum(1 for m in valid_m if m.pct_change > 0) if valid_m else 0
        consensus_ratio = up_count / len(valid_m) if valid_m else 0

        norm_pct = min(max(avg_pct / 20, -1), 1)
        pct_score = (norm_pct + 1) / 2 * 25
        consensus_score = consensus_ratio * 30

        if best_s:
            norm_ret = min(max(best_s["total_return"] * 100 / 30, -1), 1)
            ret_score = (norm_ret + 1) / 2 * 25
            norm_sharpe = min(max(best_s["sharpe"] / 2, -1), 1)
            sharpe_score = (norm_sharpe + 1) / 2 * 20
        else:
            ret_score = 0
            sharpe_score = 0

        total_score = consensus_score + pct_score + ret_score + sharpe_score

        if consensus_ratio >= 0.6 and avg_pct > 3 and best_s and best_s["total_return"] > 0 and best_s["sharpe"] > 0.5:
            tag = "强烈推荐"
        elif consensus_ratio >= 0.5 and avg_pct > 1 and best_s and best_s["total_return"] > 0:
            tag = "值得关注"
        elif avg_pct > 0 and best_s and best_s["total_return"] > 0:
            tag = "谨慎关注"
        else:
            tag = "暂不推荐"

        all_results.append({
            "code": code,
            "name": data["name"],
            "market": data["market"],
            "current_price": data["current_price"],
            "avg_pct_change": avg_pct,
            "consensus": data["consensus"],
            "model_count": len(valid_m),
            "best_strategy": best_s["name"] if best_s else "N/A",
            "best_return": best_s["total_return"] if best_s else 0,
            "best_sharpe": best_s["sharpe"] if best_s else 0,
            "score": total_score,
            "tag": tag,
        })

        report = generate_report(
            data["name"], code, data["market"],
            data["models"], [],
            data["current_price"], data["risk"], steps,
        )
        all_reports.append(report)
        (OUTPUT_DIR / f"{code}_report.txt").write_text(report)

    all_results.sort(key=lambda x: x["score"], reverse=True)

    # 生成汇总
    summary = _build_summary(all_results, failed, added, steps, len(model_names), len(strats))

    (OUTPUT_DIR / "summary.txt").write_text(summary)
    (OUTPUT_DIR / "all_reports.txt").write_text("\n\n---\n\n".join(all_reports))
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return all_results, summary, "\n\n---\n\n".join(all_reports)


def _build_summary(results, failed, added, steps, model_count, strat_count):
    recommended = [r for r in results if r["tag"] in ("强烈推荐", "值得关注")]
    cautious = [r for r in results if r["tag"] == "谨慎关注"]
    not_rec = [r for r in results if r["tag"] == "暂不推荐"]

    lines = [
        f"# 批量扫描报告 — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"自选导入: {added}只新增 | 扫描: {len(results)}只成功 | 失败: {len(failed)}只",
        f"模型: {model_count}个全量 | 策略: {strat_count}种 | 预测{steps}天",
        "",
        "## 排名汇总",
        "",
        "| # | 代码 | 名称 | 现价 | 涨幅预测 | 共识 | 模型数 | 最佳策略 | 收益% | 夏普 | 评分 | 评级 |",
        "|---|------|------|------|---------|------|-------|---------|-------|------|------|------|",
    ]

    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {r['code']} | {r['name']} | {r['current_price']:.2f} | "
            f"{r['avg_pct_change']:+.1f}% | {r['consensus']} | {r['model_count']} | "
            f"{r['best_strategy']} | {r['best_return']*100:+.1f}% | {r['best_sharpe']:.2f} | "
            f"{r['score']:.1f} | {r['tag']} |"
        )

    lines.append("")
    lines.append("## 强烈推荐 + 值得关注")
    if recommended:
        for r in recommended:
            lines.append(f"- **{r['name']}** ({r['market']}:{r['code']}): "
                         f"涨幅{r['avg_pct_change']:+.1f}% 共识{r['consensus']} "
                         f"策略{r['best_strategy']} "
                         f"收益{r['best_return']*100:+.1f}% 夏普{r['best_sharpe']:.2f}")
    else:
        lines.append("- 暂无")

    if cautious:
        lines.append("")
        lines.append("## 谨慎关注")
        for r in cautious:
            lines.append(f"- {r['name']} ({r['code']}): "
                         f"涨幅{r['avg_pct_change']:+.1f}% 评分{r['score']:.1f}")

    if not_rec:
        lines.append("")
        lines.append("## 暂不推荐")
        for r in not_rec:
            lines.append(f"- {r['name']} ({r['code']}): "
                         f"涨幅{r['avg_pct_change']:+.1f}% 评分{r['score']:.1f}")

    if failed:
        lines.append("")
        lines.append("## 数据获取失败")
        for code, name, market, err in failed:
            lines.append(f"- {name} ({market}:{code}): {err}")

    return "\n".join(lines)
