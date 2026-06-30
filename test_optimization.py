#!/usr/bin/env python3
"""
快速测试：交叉验证 + 超参数优化
"""

import sys
import pandas as pd
import numpy as np

sys.path.insert(0, "/Users/yjwang/stock-prediction")

from src.models.gbdt import GBDTModel
from src.models.xgb import XGBoostModel
from src.models.optimization import GBDTTuner, XGBoostTuner, get_tuner


def generate_test_data(n=500):
    """生成测试数据"""
    dates = pd.date_range(end="2024-01-01", periods=n, freq="D")
    
    price = 100.0
    prices = []
    for _ in range(n):
        change = np.random.normal(0, 0.02)
        price *= (1 + change)
        prices.append(price)
    
    df = pd.DataFrame({
        "Date": dates,
        "Open": prices,
        "High": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        "Low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        "Close": prices,
        "Volume": np.random.randint(1000000, 10000000, n),
    })
    return df


def test_cross_validation():
    """测试交叉验证"""
    print("\n" + "="*60)
    print("测试交叉验证")
    print("="*60)
    
    df = generate_test_data(500)
    model = GBDTModel()
    
    result = model.run(df, steps=10, use_cv=True, n_splits=3)
    
    print("\n测试集指标:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")
    
    if result.cv_result:
        print("\n交叉验证平均指标:")
        for k, v in result.cv_result.avg_metrics.items():
            std = result.cv_result.std_metrics[k]
            print(f"  {k}: {v:.4f} ± {std:.4f}")
    
    return result


def test_hyperparameter_tuning():
    """测试超参数优化"""
    print("\n" + "="*60)
    print("测试超参数优化（GBDT）")
    print("="*60)
    
    df = generate_test_data(300)
    
    tuner = GBDTTuner(n_trials=10, n_splits=3)  # 少试几次，快速测试
    result = tuner.tune(df)
    
    print(f"\n最佳参数: {result.best_params}")
    print(f"最佳分数: {result.best_score:.4f}")
    print(f"优化时间: {result.optimization_time:.2f}s")
    
    return result


def test_xgb_tuning():
    """测试XGBoost调优"""
    print("\n" + "="*60)
    print("测试超参数优化（XGBoost）")
    print("="*60)
    
    df = generate_test_data(300)
    
    tuner = XGBoostTuner(n_trials=10, n_splits=3)
    result = tuner.tune(df)
    
    print(f"\n最佳参数: {result.best_params}")
    print(f"最佳分数: {result.best_score:.4f}")
    
    return result


def main():
    import time
    
    start = time.time()
    
    # 测试1: 交叉验证
    cv_result = test_cross_validation()
    
    # 测试2: GBDT调优
    gbdt_tune = test_hyperparameter_tuning()
    
    # 测试3: XGBoost调优
    xgb_tune = test_xgb_tuning()
    
    elapsed = time.time() - start
    
    print("\n" + "="*60)
    print(f"所有测试完成！总耗时: {elapsed:.2f}s")
    print("="*60)


if __name__ == "__main__":
    main()