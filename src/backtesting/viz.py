import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from .models import BacktestResult


def plot_backtest(result: BacktestResult, save_path: str | None = None,
                  title: str = "回测结果"):
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    dates = result.equity_curve.index

    ax1 = axes[0]
    ax1.plot(dates, result.equity_curve, label="组合净值", color="black", linewidth=1)
    ax1.axhline(y=result.config.initial_capital, color="gray", linestyle="--",
                linewidth=0.8, label=f"初始资金 {result.config.initial_capital:.0f}")
    for t in result.trades:
        if t.exit_date is None:
            continue
        color = "green" if t.pnl > 0 else "red"
        marker = "^" if t.pnl > 0 else "v"
        eq = result.equity_curve[dates == t.exit_date]
        if not eq.empty:
            ax1.scatter(t.exit_date, eq.iloc[0],
                        color=color, marker=marker, s=60, zorder=5)
    ax1.set_ylabel("净值")
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    peak = np.maximum.accumulate(result.equity_curve.values)
    dd = (result.equity_curve.values - peak) / peak * 100
    ax2.fill_between(dates, dd, 0, color="red", alpha=0.3)
    ax2.plot(dates, dd, color="red", linewidth=0.8)
    ax2.set_ylabel("回撤 (%)")
    ax2.grid(True, alpha=0.3)

    # Holdings
    ax3 = axes[2]
    ax3.fill_between(dates, result.holdings_curve.values, alpha=0.4, label="持仓市值")
    ax3.plot(dates, result.holdings_curve.values, linewidth=0.8)
    ax3.set_ylabel("持仓市值")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"回测图已保存: {save_path}")
    plt.close()
