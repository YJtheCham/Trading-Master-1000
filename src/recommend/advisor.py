"""
LLM 策略顾问: DeepSeek API (OpenAI 兼容) + stock-skill 视角
"""
import logging, json, urllib3
from typing import Optional

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是量化分析助手, 综合技术/宏观/基本面三维度分析交易策略。

数据报告包含风控指标(VaR/CVaR/夏普/回撤)、5个模型预测结果、8个策略回测对比。

按以下格式输出, 总共300字以内:

**📊 技术面(恨铁):** [基于模型共识和回测数据的客观判断, 语气萌但结论硬]

**🌍 宏观面(TraderS):** [基于行业/市场环境的推断, 标注"推测"]

**🔗 基本面(Serenity):** [基于行业地位的推断, 标注"推测"]

**综合建议:** 一句话指出最值得关注的策略和风险"""


def call_llm(report: str, api_key: str,
             base_url: str = "https://api.deepseek.com/v1",
             model: str = "deepseek-chat",
             timeout: int = 60,
             system_prompt: str = SYSTEM_PROMPT) -> Optional[str]:
    """调用 DeepSeek API (OpenAI 兼容格式)"""
    import requests

    if not api_key:
        return None

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": report},
        ],
        "temperature": 0.7,
        "max_tokens": 800,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout,
                             verify=False)
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices") or []
            if not choices or not choices[0].get("message"):
                logger.warning(f"LLM 返回空 choices: {json.dumps(data)[:200]}")
                return None
            return choices[0]["message"].get("content", "")
        else:
            logger.warning(f"LLM API 返回 {resp.status_code}: {resp.text[:200]}")
            return f"LLM 调用失败 (HTTP {resp.status_code})"
    except requests.exceptions.Timeout:
        return "LLM 调用超时, 请在设置里确认 API 可连通"
    except Exception as e:
        logger.warning(f"LLM 调用异常: {e}")
        return f"LLM 调用异常: {e}"


def analyze_with_llm(report: str) -> Optional[str]:
    """使用配置的 LLM 分析报告 (三人合议)"""
    from src.utils.config import get_llm_config
    cfg = get_llm_config()
    if not cfg["api_key"]:
        return None
    enriched = report + "\n请从技术面(恨铁)、宏观面(TraderS)、基本面(Serenity)三维度分析, 给出综合策略建议。"
    return call_llm(enriched, cfg["api_key"], cfg["base_url"], cfg["model"])
