"""
全功能演示脚本
运行: python3 demo.py
模拟离线数据展示所有模型、风控指标、CLI 功能
"""
import subprocess, sys, os
import numpy as np
import pandas as pd

def log(msg="", **kwargs):
    print(msg, flush=True, **kwargs)

log(f"{'='*60}")
log(f"  自选股价预测 & 风控系统 — 功能演示")
log(f"{'='*60}")

np.random.seed(42)
N = 250
dates = pd.date_range("2024-01-01", periods=N, freq="B")
prices = 100 + np.cumsum(np.random.randn(N) * 0.8 + 0.05)
volume = np.random.randint(5e6, 5e8, N)
df = pd.DataFrame({"Date": dates, "Close": prices, "Volume": volume})
log(f"模拟数据: {len(df)} 个交易日 [{df['Close'].min():.2f} ~ {df['Close'].max():.2f}]\n")

# ─── 1. 风控指标 ──────────────────────────────────────────
log("【1】风控指标")
from src.risk.metrics import calc_all_risk_metrics
risk = calc_all_risk_metrics(df)
for k, v in risk.items():
    log(f"  {k:>15s} : {v}")
log()

# ─── 2. 传统模型预测 ───────────────────────────────────────
log("【2】传统模型预测对比 (未来15天)")
from src.models.factory import run_models
results = run_models(df, ["arima", "gbdt", "xgboost"], steps=15)
log(f"  {'模型':<10s} {'MAE':>8s} {'RMSE':>8s} {'MAPE(%)':>8s} {'方向':>8s} {'预测末价':>10s}")
log(f"  {'-'*52}")
for name, r in results.items():
    m = r.metrics
    direction = "↑涨" if r.forecast[-1] > r.history[-1] else "↓跌"
    log(f"  {name:<10s} {str(m.get('MAE','-')):>8s} {str(m.get('RMSE','-')):>8s} "
        f"{str(m.get('MAPE','-')):>8s} {direction:>8s} {r.forecast[-1]:>8.2f}")
log()

# ─── 3. Transformer (独立进程) ─────────────────────────────
log("【3】Transformer 模型预测 (PyTorch, 未来10天)")
code = """
import numpy as np, pandas as pd
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(250) * 0.8 + 0.05)
df = pd.DataFrame({'Date': pd.date_range('2024-01-01', periods=250, freq='B'), 'Close': prices, 'Volume': np.random.randint(5e6,5e8,250)})
from src.models.transformer_model import TransformerModel
m = TransformerModel(lookback=20, epochs=15)
r = m.run(df, steps=10)
direction = '↑涨' if r.forecast[-1] > r.history[-1] else '↓跌'
print(f'Transformer: MAPE={r.metrics[\"MAPE\"]}% 方向={direction} 预测末价={r.forecast[-1]:.2f}')
"""
result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                        cwd=os.path.dirname(__file__))
log(f"  {result.stdout.strip()}")
log()

# ─── 4. 模型集成 ───────────────────────────────────────────
log("【4】多模型集成投票")
end_prices = {name: r.forecast[-1] for name, r in results.items()}
directions = [r.forecast[-1] > r.history[-1] for r in results.values()]
vote = "↑涨" if sum(directions) > len(directions) / 2 else "↓跌"
log(f"  模型: {', '.join(results.keys())}")
log(f"  平均预测末价: {np.mean(list(end_prices.values())):.2f}")
log(f"  投票决议: {vote}")
log()

# ─── 5. CLI 命令 ───────────────────────────────────────────
log("【5】CLI 命令演示")
def cli(cmd):
    r = subprocess.run([sys.executable, "-m", "src.cli.main"] + cmd.split(),
                       capture_output=True, text=True, cwd=os.path.dirname(__file__))
    return r.stdout.strip()

log("  $ stock models")
log(f"  {cli('models')}")
log()
log("  $ stock add 600519 A -n 贵州茅台")
log(f"  {cli('add 600519 A -n 贵州茅台')}")
log("  $ stock add AAPL US -n Apple")
log(f"  {cli('add AAPL US -n Apple')}")
log("  $ stock list")
log(f"  {cli('list')}")
log()

# ─── 6. 预测图 ─────────────────────────────────────────────
log("【6】预测可视化")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("股价预测演示 — 多模型对比", fontsize=14, fontweight="bold")

ax1 = axes[0]
ax1.plot(dates, prices, label="历史收盘价", color="black", linewidth=1)
colors = {"arima": "#E74C3C", "gbdt": "#2ECC71", "xgboost": "#F39C12"}
for name, r in results.items():
    ax1.plot(r.forecast_dates, r.forecast, label=f"{name}",
             color=colors.get(name, "purple"), linestyle="--", marker=".", linewidth=1.5)
ax1.set_ylabel("价格")
ax1.legend(fontsize=10, ncol=4)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.bar(dates, volume / 1e8, width=1, color="steelblue", alpha=0.5)
ax2.set_ylabel("成交量(亿)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
path = "demo_prediction.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
log(f"  预测图已保存: {path}")

# ─── 总结 ──────────────────────────────────────────────────
log(f"\n{'='*60}")
log("  演示完成 ✅")
log("  已展示: 风控 | ARIMA | GBDT | XGBoost | Transformer | CLI | 可视化")
log(f"{'='*60}")
