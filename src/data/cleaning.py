"""
数据清洗: 缺失值处理 / 异常值检测 / 格式标准化

用法:
  df = clean_market_data(raw_df)
  自动修复常见问题
"""
import numpy as np
import pandas as pd


REQUIRED_COLS = {"Date", "Open", "Close", "High", "Low", "Volume"}


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """全自动清洗入口: 标准化列名 → 排序去重 → 缺失值 → 异常值"""
    if df.empty:
        return df

    df = df.copy()

    # 1. 列名标准化
    df = _normalize_columns(df)

    # 2. 日期排序 + 去重
    df = _deduplicate_sort(df)

    # 3. 数值列转换
    for col in ["Open", "Close", "High", "Low", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. 缺失值处理
    df = _fill_missing(df)

    # 5. 异常值检测
    df = _remove_outliers(df)

    # 6. 衍生指标
    df = _add_derived(df)

    return df.reset_index(drop=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名"""
    mapping = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("date", "trade_date", "datetime", "日期", "交易日期"):
            mapping[c] = "Date"
        elif cl in ("open", "开盘"):
            mapping[c] = "Open"
        elif cl in ("close", "收盘", "收盘价"):
            mapping[c] = "Close"
        elif cl in ("high", "最高", "最高价"):
            mapping[c] = "High"
        elif cl in ("low", "最低", "最低价"):
            mapping[c] = "Low"
        elif cl in ("volume", "vol", "成交量", "成交数量"):
            mapping[c] = "Volume"
        elif cl in ("amount", "成交额", "成交金额"):
            mapping[c] = "Amount"
    return df.rename(columns=mapping)


def _deduplicate_sort(df: pd.DataFrame) -> pd.DataFrame:
    """按日期排序 + 去重重复日期"""
    if "Date" not in df.columns:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """缺失值填充: OHLC 向前填充, Volume 填 0"""
    for col in ["Open", "Close", "High", "Low"]:
        if col in df.columns:
            df[col] = df[col].ffill()
            # 如果开头还有缺失, 用第一个有效值填充
            if df[col].isna().any():
                first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else 0
                df[col] = df[col].fillna(first_valid)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0).clip(lower=0)
    return df


def _remove_outliers(df: pd.DataFrame, z_thresh: float = 5.0) -> pd.DataFrame:
    """异常值检测: z-score 方法, 超出阈值用前后均值替换"""
    for col in ["Open", "Close", "High", "Low"]:
        if col not in df.columns or len(df) < 20:
            continue
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            continue
        z = (df[col] - mean).abs() / std
        outliers = z > z_thresh
        if outliers.any():
            # 用前一个和后一个的平均值替换
            for idx in df[outliers].index:
                before = df[col].iloc[max(0, idx - 1)]
                after = df[col].iloc[min(len(df) - 1, idx + 1)]
                df.loc[idx, col] = (before + after) / 2
    return df


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """添加衍生指标"""
    if "Close" in df.columns and len(df) > 1:
        df["Return"] = df["Close"].pct_change()
    return df


def validate_data(df: pd.DataFrame) -> list[str]:
    """数据质量检查, 返回问题列表"""
    issues = []
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        issues.append(f"缺少列: {missing}")

    if "Date" in df.columns and len(df) > 1:
        date_range = (df["Date"].max() - df["Date"].min()).days
        coverage = len(df) / max(date_range, 1)
        if coverage < 0.3:
            issues.append(f"数据稀疏 ({coverage:.0%} 的交易日有数据)")

    for col in ["Close", "Volume"]:
        if col in df.columns and df[col].isna().sum() > 0:
            issues.append(f"{col} 有 {df[col].isna().sum()} 个缺失值")

    if "Close" in df.columns:
        neg_count = (df["Close"] <= 0).sum()
        if neg_count > 0:
            issues.append(f"收盘价有 {neg_count} 个非正值")

    return issues
