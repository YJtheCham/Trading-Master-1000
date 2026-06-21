"""
超参数优化模块
支持 Optuna (TPE) 和 Sklearn RandomizedSearchCV 两种优化方式
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    best_trial: int
    n_trials: int
    optimization_time: float
    all_trials: List[Dict[str, Any]] = field(default_factory=list)
    metric_name: str = "rmse"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算MAPE"""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


@dataclass
class ModelTuner:
    """模型超参数调优器基类"""
    model_name: str
    metric: str = "rmse"  # rmse, mae, mape
    n_trials: int = 50
    n_splits: int = 5
    random_state: int = 42
    cache_dir: str = "data/cache/tuning"
    
    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.metric == "rmse":
            self.scoring = make_scorer(rmse, greater_is_better=False)
        elif self.metric == "mae":
            self.scoring = make_scorer(mean_absolute_error, greater_is_better=False)
        elif self.metric == "mape":
            self.scoring = make_scorer(mape, greater_is_better=False)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _get_cache_path(self, df: pd.DataFrame) -> Path:
        """获取缓存文件路径"""
        df_hash = hash(str(df.values.tobytes()))
        return self.cache_dir / f"{self.model_name}_{self.metric}_{self.n_trials}_{abs(df_hash)}.json"
    
    def _load_cache(self, df: pd.DataFrame) -> Optional[OptimizationResult]:
        """加载缓存的优化结果"""
        cache_path = self._get_cache_path(df)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                return OptimizationResult(**data)
            except Exception:
                return None
        return None
    
    def _save_cache(self, result: OptimizationResult, df: pd.DataFrame):
        """保存优化结果到缓存"""
        cache_path = self._get_cache_path(df)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({
                "best_params": result.best_params,
                "best_score": result.best_score,
                "best_trial": result.best_trial,
                "n_trials": result.n_trials,
                "optimization_time": result.optimization_time,
                "all_trials": result.all_trials,
                "metric_name": result.metric_name
            }, f, indent=2)
    
    @abstractmethod
    def get_param_space(self) -> Dict[str, Any]:
        """返回参数搜索空间"""
        pass
    
    @abstractmethod
    def create_model(self, **params):
        """根据参数创建模型"""
        pass
    
    def evaluate_params(self, params: Dict[str, Any], df: pd.DataFrame) -> float:
        """评估一组参数"""
        from sklearn.model_selection import TimeSeriesSplit
        
        from .features import engineer_features
        
        feat_df = engineer_features(df)
        n = len(feat_df)
        
        if n < 100:
            return float('inf')
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = []
        
        for train_idx, test_idx in tscv.split(feat_df):
            train_df = feat_df.iloc[train_idx]
            test_df = feat_df.iloc[test_idx]
            
            model = self.create_model(**params)
            try:
                model.train(train_df)
                pred = model.predict(len(test_idx))
                
                actual = test_df["Close"].values
                pred = pred[:len(actual)]
                
                if self.metric == "rmse":
                    score = rmse(actual, pred)
                elif self.metric == "mae":
                    score = mean_absolute_error(actual, pred)
                elif self.metric == "mape":
                    score = mape(actual, pred)
                else:
                    score = rmse(actual, pred)
                
                scores.append(score)
            except Exception as e:
                scores.append(float('inf'))
        
        return np.mean(scores)
    
    def tune(self, df: pd.DataFrame, use_cache: bool = True) -> OptimizationResult:
        """执行超参数优化"""
        if use_cache:
            cached = self._load_cache(df)
            if cached is not None:
                print(f"Loaded cached result from {self._get_cache_path(df)}")
                return cached
        
        print(f"Starting hyperparameter optimization for {self.model_name}...")
        print(f"Metric: {self.metric}, Trials: {self.n_trials}")
        
        import time
        start_time = time.time()
        
        best_score = float('inf')
        best_params = {}
        best_trial = 0
        all_trials = []
        
        param_space = self.get_param_space()
        
        for trial in range(self.n_trials):
            params = {}
            for param_name, param_config in param_space.items():
                param_type = param_config["type"]
                
                if param_type == "int":
                    low, high = param_config["range"]
                    params[param_name] = np.random.randint(low, high + 1)
                elif param_type == "float":
                    low, high = param_config["range"]
                    if param_config.get("log", False):
                        params[param_name] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                    else:
                        params[param_name] = np.random.uniform(low, high)
                elif param_type == "choice":
                    params[param_name] = np.random.choice(param_config["values"])
            
            score = self.evaluate_params(params, df)
            
            trial_result = {
                "trial": trial,
                "params": params,
                "score": score
            }
            all_trials.append(trial_result)
            
            if score < best_score:
                best_score = score
                best_params = params
                best_trial = trial
                print(f"Trial {trial}: New best! Score={score:.4f}, Params={params}")
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_trial=best_trial,
            n_trials=self.n_trials,
            optimization_time=optimization_time,
            all_trials=all_trials,
            metric_name=self.metric
        )
        
        self._save_cache(result, df)
        
        print(f"\nOptimization complete!")
        print(f"Best score: {best_score:.4f}")
        print(f"Best params: {best_params}")
        print(f"Time: {optimization_time:.2f}s")
        
        return result


class GBDTTuner(ModelTuner):
    """GBDT 超参数调优器"""
    
    def __init__(self, **kwargs):
        super().__init__("gbdt", **kwargs)
    
    def get_param_space(self) -> Dict[str, Any]:
        return {
            "n_estimators": {
                "type": "int",
                "range": [100, 500]
            },
            "learning_rate": {
                "type": "float",
                "range": [0.01, 0.1],
                "log": True
            },
            "max_depth": {
                "type": "int",
                "range": [4, 10]
            },
            "subsample": {
                "type": "float",
                "range": [0.6, 1.0]
            },
            "min_samples_split": {
                "type": "int",
                "range": [2, 20]
            },
            "min_samples_leaf": {
                "type": "int",
                "range": [1, 10]
            }
        }
    
    def create_model(self, **params):
        from .gbdt import GBDTModel
        
        params.setdefault("random_state", self.random_state)
        model = GBDTModel()
        model._tuned_params = params
        return model


class XGBoostTuner(ModelTuner):
    """XGBoost 超参数调优器"""
    
    def __init__(self, **kwargs):
        super().__init__("xgboost", **kwargs)
    
    def get_param_space(self) -> Dict[str, Any]:
        return {
            "n_estimators": {
                "type": "int",
                "range": [200, 800]
            },
            "learning_rate": {
                "type": "float",
                "range": [0.01, 0.2],
                "log": True
            },
            "max_depth": {
                "type": "int",
                "range": [5, 10]
            },
            "subsample": {
                "type": "float",
                "range": [0.6, 1.0]
            },
            "colsample_bytree": {
                "type": "float",
                "range": [0.6, 1.0]
            },
            "reg_alpha": {
                "type": "float",
                "range": [0.0, 1.0]
            },
            "reg_lambda": {
                "type": "float",
                "range": [0.5, 2.0]
            }
        }
    
    def create_model(self, **params):
        from .xgb import XGBoostModel
        
        params.setdefault("random_state", self.random_state)
        params.setdefault("verbosity", 0)
        model = XGBoostModel()
        model._tuned_params = params
        return model


class LSTMTuner(ModelTuner):
    """LSTM 超参数调优器"""
    
    def __init__(self, **kwargs):
        super().__init__("lstm", **kwargs)
        self.n_trials = kwargs.get("n_trials", 30)
    
    def get_param_space(self) -> Dict[str, Any]:
        return {
            "lookback": {
                "type": "int",
                "range": [30, 90]
            },
            "hidden_size": {
                "type": "int",
                "range": [32, 128]
            },
            "num_layers": {
                "type": "choice",
                "values": [1, 2, 3]
            },
            "dropout": {
                "type": "float",
                "range": [0.1, 0.4]
            },
            "learning_rate": {
                "type": "float",
                "range": [1e-4, 5e-3],
                "log": True
            },
            "batch_size": {
                "type": "choice",
                "values": [16, 32, 64]
            }
        }
    
    def create_model(self, **params):
        from .lstm_model import LSTMModel
        
        params.setdefault("epochs", 50)
        model = LSTMModel(
            lookback=params.pop("lookback", 60),
            hidden=params.pop("hidden_size", 64),
            lr=params.pop("learning_rate", 1e-3),
            batch_size=params.pop("batch_size", 32),
            dropout=params.pop("dropout", 0.2)
        )
        model._tuned_params = params
        return model


class TransformerTuner(ModelTuner):
    """Transformer 超参数调优器"""
    
    def __init__(self, **kwargs):
        super().__init__("transformer", **kwargs)
        self.n_trials = kwargs.get("n_trials", 30)
    
    def get_param_space(self) -> Dict[str, Any]:
        return {
            "lookback": {
                "type": "int",
                "range": [30, 90]
            },
            "d_model": {
                "type": "choice",
                "values": [64, 128, 256]
            },
            "nhead": {
                "type": "choice",
                "values": [4, 8]
            },
            "num_layers": {
                "type": "choice",
                "values": [2, 3, 4]
            },
            "dim_feedforward": {
                "type": "int",
                "range": [64, 256]
            },
            "dropout": {
                "type": "float",
                "range": [0.1, 0.3]
            },
            "learning_rate": {
                "type": "float",
                "range": [1e-4, 5e-3],
                "log": True
            }
        }
    
    def create_model(self, **params):
        from .transformer_model import TransformerModel
        
        params.setdefault("epochs", 40)
        model = TransformerModel(
            lookback=params.pop("lookback", 60),
            d_model=params.pop("d_model", 64),
            nhead=params.pop("nhead", 4),
            num_layers=params.pop("num_layers", 3),
            dim_feedforward=params.pop("dim_feedforward", 128),
            dropout=params.pop("dropout", 0.1),
            lr=params.pop("learning_rate", 1e-3)
        )
        model._tuned_params = params
        return model


TUNER_REGISTRY = {
    "gbdt": GBDTTuner,
    "xgboost": XGBoostTuner,
    "lstm": LSTMTuner,
    "transformer": TransformerTuner,
}


def get_tuner(model_name: str, **kwargs) -> ModelTuner:
    """获取指定模型的调优器"""
    tuner_cls = TUNER_REGISTRY.get(model_name)
    if tuner_cls is None:
        raise ValueError(f"Unknown model: {model_name}")
    return tuner_cls(**kwargs)


def tune_model(model_name: str, df: pd.DataFrame, **kwargs) -> OptimizationResult:
    """调优指定模型"""
    tuner = get_tuner(model_name, **kwargs)
    return tuner.tune(df)