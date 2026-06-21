from .base import BaseModel, PredictionResult
from .arima import ArimaModel
from .gbdt import GBDTModel
from .xgb import XGBoostModel

import numpy as np
import pandas as pd


# 懒加载 PyTorch 模型 (避免 Streamlit fork 冲突)
def _get_registry() -> dict[str, type[BaseModel]]:
    registry = {
        "arima": ArimaModel,
        "gbdt": GBDTModel,
        "xgboost": XGBoostModel,
    }
    try:
        import sys
        if "torch" not in sys.modules:
            import torch
            torch.set_num_threads(1)
        from .lstm_model import LSTMModel
        registry["lstm"] = LSTMModel
    except Exception:
        pass
    try:
        from .transformer_model import TransformerModel
        registry["transformer"] = TransformerModel
    except Exception:
        pass
    return registry


MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def _init_registry():
    global MODEL_REGISTRY
    if not MODEL_REGISTRY:
        MODEL_REGISTRY.update(_get_registry())


def list_models() -> list[str]:
    _init_registry()
    return list(MODEL_REGISTRY.keys())


def run_models(df: pd.DataFrame, model_names: list[str] | None = None,
               steps: int = 30, data_source: str = "") -> dict[str, PredictionResult]:
    _init_registry()
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    results = {}
    for name in model_names:
        cls = MODEL_REGISTRY.get(name)
        if cls is None:
            continue
        model = cls()
        model._data_source = data_source
        try:
            results[name] = model.run(df, steps=steps)
        except Exception as e:
            results[name] = PredictionResult(
                model_name=name,
                forecast=np.array([]),
                history=np.array([]),
                dates=[],
                forecast_dates=[],
                metrics={"error": str(e)},
                data_source=data_source,
            )
    return results
