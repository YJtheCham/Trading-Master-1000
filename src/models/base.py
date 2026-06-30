from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union, List, Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class CrossValidationResult:
    """交叉验证结果"""
    model_name: str
    n_splits: int
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    avg_metrics: Dict[str, float] = field(default_factory=dict)
    std_metrics: Dict[str, float] = field(default_factory=dict)
    fold_predictions: List[np.ndarray] = field(default_factory=list)
    fold_actuals: List[np.ndarray] = field(default_factory=list)


@dataclass
class PredictionResult:
    model_name: str
    forecast: np.ndarray
    history: np.ndarray
    dates: list
    forecast_dates: list
    metrics: dict
    cv_result: Optional[CrossValidationResult] = None
    model_params: Dict[str, Any] = field(default_factory=dict)
    data_source: str = ""
    feature_names: List[str] = field(default_factory=list)


class BaseModel(ABC):
    """模型基类: 支持单变量(Close)和多维特征两种模式"""

    def __init__(self, name: str):
        self.name = name
        self._tuned_params: Dict[str, Any] = {}
        self._data_source: str = ""
        self._feature_names: List[str] = []

    @abstractmethod
    def train(self, data: Union[pd.DataFrame, np.ndarray]):
        """训练模型. data 为特征 DataFrame (多维模型) 或 Price Series (单变量)."""
        ...

    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """预测未来 steps 天的价格."""
        ...

    def get_param_info(self) -> Dict[str, Any]:
        """返回模型参数信息（子类可覆盖追加）"""
        return {
            "model": self.name,
            "params": dict(self._tuned_params) if self._tuned_params else {},
            "data_source": self._data_source,
            "features": list(self._feature_names) if self._feature_names else [],
        }

    def run(self, df: pd.DataFrame, steps: int = 30,
            test_ratio: float = 0.2, use_cv: bool = False, 
            n_splits: int = 5) -> PredictionResult:
        from .features import engineer_features

        feat_df = engineer_features(df)
        n = len(feat_df)

        cv_result = None
        if use_cv:
            print(f"\n{'='*50}")
            print(f"执行交叉验证: {self.name}")
            print(f"{'='*50}")
            cv_result = self.cross_validate(feat_df, n_splits=n_splits)
            print(f"\n交叉验证完成，继续训练...")
            print(f"{'='*50}\n")

        train_size = int(n * (1 - test_ratio))
        train_feat = feat_df.iloc[:train_size]
        test_target = feat_df["Close"].values[train_size + 1:]  # 对齐: 预测Close[t+1]

        self.train(train_feat)
        test_forecast = self.predict(n - train_size)
        actual_len = min(len(test_target), len(test_forecast))
        metrics = self._calc_metrics(test_target[:actual_len],
                                     test_forecast[:actual_len])

        self.train(feat_df)
        forecast = self.predict(steps)

        dates = df["Date"].tolist()
        last_date = pd.Timestamp(dates[-1])
        forecast_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(steps)]

        return PredictionResult(
            model_name=self.name,
            forecast=forecast,
            history=feat_df["Close"].values if "Close" in feat_df.columns else np.array([]),
            dates=dates,
            forecast_dates=forecast_dates,
            metrics=metrics,
            cv_result=cv_result,
            model_params=self.get_param_info(),
            data_source=self._data_source,
            feature_names=list(self._feature_names) if self._feature_names else list(self._x_cols) if hasattr(self, '_x_cols') and self._x_cols else [],
        )

    def _calc_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
        
        direction_actual = np.sign(actual[1:] - actual[:-1])
        direction_pred = np.sign(predicted[1:] - predicted[:-1])
        direction_accuracy = np.mean(direction_actual == direction_pred) * 100
        
        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 2),
            "Direction_Accuracy": round(direction_accuracy, 2)
        }

    def cross_validate(self, df: pd.DataFrame, n_splits: int = 5, 
                       use_cache: bool = True) -> CrossValidationResult:
        """时序交叉验证（带缓存优化）"""
        from .features import engineer_features
        import hashlib
        import pickle
        from pathlib import Path
        
        cache_dir = Path("data/cache/cv")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        feat_df = engineer_features(df)
        n = len(feat_df)
        
        if n < 100:
            raise ValueError(f"数据量不足，至少需要100条数据，当前只有{n}条")
        
        df_hash = hashlib.md5(str(feat_df.values.tobytes()).encode()).hexdigest()[:8]
        cache_file = cache_dir / f"{self.name}_{n_splits}_{df_hash}.pkl"
        
        if use_cache and cache_file.exists():
            print(f"加载缓存的交叉验证结果: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics = []
        fold_predictions = []
        fold_actuals = []
        
        print(f"\n执行 {n_splits} 折时序交叉验证...")
        print(f"总数据量: {n} 条")
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(feat_df)):
            print(f"\nFold {fold + 1}/{n_splits}")
            print(f"  训练集: {train_idx[0]}-{train_idx[-1]} ({len(train_idx)} 条)")
            print(f"  测试集: {test_idx[0]}-{test_idx[-1]} ({len(test_idx)} 条)")
            
            train_df = feat_df.iloc[train_idx]
            shifted_idx = np.clip(test_idx + 1, 0, n - 1)  # 预测Close[t+1]
            test_target = feat_df["Close"].values[shifted_idx]
            
            self.train(train_df)
            test_forecast = self.predict(len(test_idx))
            
            actual_len = min(len(test_target), len(test_forecast))
            test_forecast = test_forecast[:actual_len]
            test_target = test_target[:actual_len]
            
            metrics = self._calc_metrics(test_target, test_forecast)
            fold_metrics.append(metrics)
            fold_predictions.append(test_forecast)
            fold_actuals.append(test_target)
            
            print(f"  MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, "
                  f"MAPE: {metrics['MAPE']:.2f}%, DirAcc: {metrics['Direction_Accuracy']:.2f}%")
        
        avg_metrics = {}
        std_metrics = {}
        
        for key in fold_metrics[0].keys():
            values = [m[key] for m in fold_metrics]
            avg_metrics[key] = round(np.mean(values), 4)
            std_metrics[key] = round(np.std(values), 4)
        
        print(f"\n交叉验证完成！平均指标:")
        for key, avg_val in avg_metrics.items():
            std_val = std_metrics[key]
            print(f"  {key}: {avg_val:.4f} ± {std_val:.4f}")
        
        result = CrossValidationResult(
            model_name=self.name,
            n_splits=n_splits,
            fold_metrics=fold_metrics,
            avg_metrics=avg_metrics,
            std_metrics=std_metrics,
            fold_predictions=fold_predictions,
            fold_actuals=fold_actuals
        )
        
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        
        return result
