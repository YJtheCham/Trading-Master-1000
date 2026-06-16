import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

WATCHLIST_FILE = DATA_DIR / "watchlist.json"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = DATA_DIR / "config.json"


class StockItem(BaseModel):
    symbol: str
    market: str
    name: str = ""
    active: bool = True


# ─── Token / 配置管理 ────────────────────────────────────
def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))


def get_tushare_token() -> str:
    """从环境变量或配置文件获取 Tushare Token"""
    import os
    token = os.environ.get("TUSHARE_TOKEN", "")
    if token:
        return token
    cfg = load_config()
    return cfg.get("tushare_token", "")


def get_llm_key() -> str:
    """LLM API Key (DeepSeek)"""
    import os
    key = os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("LLM_API_KEY", "")
    if key:
        return key
    return load_config().get("llm_api_key", "")


def get_llm_config() -> dict:
    """LLM 配置"""
    cfg = load_config()
    return {
        "api_key": get_llm_key(),
        "base_url": cfg.get("llm_base_url", "https://api.deepseek.com/v1"),
        "model": cfg.get("llm_model", "deepseek-chat"),
    }


def get_telegram_config() -> dict:
    """Telegram Bot 配置"""
    import os
    cfg = load_config()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "") or cfg.get("telegram_token", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "") or cfg.get("telegram_chat_id", "")
    return {"token": token, "chat_id": chat_id}
