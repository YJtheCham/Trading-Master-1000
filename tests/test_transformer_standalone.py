"""单独运行: python3 tests/test_transformer_standalone.py"""
import numpy as np
import pandas as pd
from src.models.transformer_model import TransformerModel

np.random.seed(42)
n = 80
prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
df = pd.DataFrame({
    "Date": pd.date_range("2024-01-01", periods=n, freq="D"),
    "Close": prices,
    "Volume": np.random.randint(1e6, 1e8, n),
})

model = TransformerModel(lookback=20, epochs=10)
result = model.run(df, steps=5)
assert len(result.forecast) == 5, f"预期5步，得到{len(result.forecast)}"
assert result.model_name == "Transformer"
print(f"[PASS] Transformer: forecast={result.forecast.round(2)}, metrics={result.metrics}")
