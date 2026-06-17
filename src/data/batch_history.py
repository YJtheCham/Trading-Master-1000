"""
批量预测历史: JSON 文件 (存储完整简报)
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.utils.config import DATA_DIR

BATCH_HISTORY_FILE = DATA_DIR / "batch_prediction_history.json"


@dataclass
class BatchRecord:
    id: str
    predicted_at: str
    steps: int
    models: list = field(default_factory=list)
    summary: list = field(default_factory=list)  # 每只股票的摘要
    details: list = field(default_factory=list)  # 完整的预测结果


def load_batch_history() -> list[BatchRecord]:
    if not BATCH_HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(BATCH_HISTORY_FILE.read_text())
        return [BatchRecord(**r) for r in data]
    except Exception:
        return []


def save_batch_history(records: list[BatchRecord]):
    BATCH_HISTORY_FILE.write_text(
        json.dumps([r.__dict__ for r in records], ensure_ascii=False, indent=2))


def add_batch_record(steps: int, models: list, summary: list, details: list):
    record = BatchRecord(
        id=f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        predicted_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        steps=steps, models=models, summary=summary, details=details,
    )
    history = load_batch_history()
    history.append(record)
    history = history[-50:]  # 最多50条
    save_batch_history(history)
    return record
