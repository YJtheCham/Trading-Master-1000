"""
Bot Webhook Server: 接收 PushPlus 回调 + 飞书事件推送
启动: python -m src.bot.server --port 8080
"""
import json
import logging
import hashlib
import re
import time
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify

from .dispatcher import parse_message, execute_task, format_reply, AgentTask

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

app = Flask(__name__)


# ─── PushPlus 回调 ─────────────────────────────────────────
@app.route("/pushplus/callback", methods=["POST"])
def pushplus_callback():
    """PushPlus 回调: 用户在微信回复消息时触发"""
    data = request.json or {}
    content = data.get("content", "")
    source = data.get("source", "")
    user = data.get("userId", "")

    if not content:
        return jsonify({"status": "empty", "message": "消息内容为空"}), 200

    task = parse_message(content)
    if not task:
        return jsonify({
            "status": "unknown_command",
            "message": f"无法识别指令: {content}\n支持: /review /daily /analyze /pm /trader",
        }), 200

    task.source = "pushplus"
    task.sender = user

    threading.Thread(target=_run_and_reply_pushplus, args=(task,), daemon=True).start()

    return jsonify({"status": "accepted", "task_id": task.task_id, "agent": task.agent}), 200


def _run_and_reply_pushplus(task: AgentTask):
    """执行任务并通过 PushPlus 回复结果"""
    execute_task(task)
    reply = format_reply(task)

    from src.utils.config import get_pushplus_token
    token = get_pushplus_token()
    if not token:
        logger.warning("PushPlus token 未配置, 无法回复")
        return

    import requests
    try:
        requests.post("http://www.pushplus.plus/send", json={
            "token": token,
            "title": f"StockPredict Bot: {task.message[:30]}",
            "content": reply,
            "template": "markdown",
        }, timeout=10)
    except Exception as e:
        logger.error(f"PushPlus 回复失败: {e}")


# ─── 飞书事件推送 ──────────────────────────────────────────
FEISHU_VERIFY_TOKEN = ""
FEISHU_ENCRYPT_KEY = ""


def _decrypt_feishu(encrypt_str: str) -> str:
    """解密飞书加密事件 (AES-256-CBC, base64编码)
    
    飞书加密原理:
    1. SHA256(Encrypt Key) → 32字节key (AES-256)
    2. 16字节随机iv + PKCS7Padding + AES-CBC加密
    3. base64(iv + encrypted_event)
    """
    import base64
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    key = hashlib.sha256(FEISHU_ENCRYPT_KEY.encode("utf-8")).digest()
    encrypted_data = base64.b64decode(encrypt_str)
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(ciphertext) + decryptor.finalize()
    decrypted = decrypted.decode("utf-8")
    pad_len = ord(decrypted[-1])
    decrypted = decrypted[:len(decrypted) - pad_len]
    return decrypted


def _decrypt_feishu_data(data: dict) -> dict:
    """解密飞书加密请求, 返回原始数据"""
    encrypt = data.get("encrypt", "")
    if not encrypt or not FEISHU_ENCRYPT_KEY:
        return data
    try:
        decrypted_str = _decrypt_feishu(encrypt)
        return json.loads(decrypted_str)
    except Exception as e:
        logger.error(f"飞书解密失败: {e}")
        return data


@app.route("/feishu/event", methods=["POST"])
def feishu_event():
    """飞书 Bot 事件回调: 群消息接收 (v2格式, 支持加密)"""
    raw_data = request.json or {}
    print(f"[FEISHU] raw_data keys: {list(raw_data.keys())}")
    if "encrypt" in raw_data:
        print(f"[FEISHU] encrypt field present, length={len(raw_data['encrypt'])}")

    # 解密 (如果启用了Encrypt Key)
    data = _decrypt_feishu_data(raw_data)
    print(f"[FEISHU] 解密后 keys: {list(data.keys())}")

    # 1. 验证挑战 (首次配置时飞书发送 challenge)
    challenge = data.get("challenge")
    if challenge:
        return jsonify({"challenge": challenge}), 200

    # 2. 解析事件 (飞书v2格式: header + event)
    header = data.get("header", {})
    event = data.get("event", {})

    # v2格式token在header里
    token = header.get("token", "") or data.get("token", "")
    if FEISHU_VERIFY_TOKEN and token != FEISHU_VERIFY_TOKEN:
        return jsonify({"status": "invalid_token"}), 403

    # v2格式: msg_type在event.message.message_type
    message = event.get("message", {})
    msg_type = message.get("message_type", "") or event.get("msg_type", "")

    if msg_type != "text":
        print(f"[FEISHU] msg_type={msg_type}, data preview: {json.dumps(data, ensure_ascii=False)[:300]}")
        return jsonify({"status": "unsupported_type", "msg_type": msg_type}), 200

    # v2格式: content在event.message.content
    content_str = message.get("content", "{}") or event.get("content", "{}")
    try:
        content_json = json.loads(content_str)
        text = content_json.get("text", "")
    except json.JSONDecodeError:
        text = content_str

    if not text:
        return jsonify({"status": "empty"}), 200

    sender_id = event.get("sender", {}).get("sender_id", {}).get("open_id", "")
    chat_id = message.get("chat_id", "")

    print(f"[FEISHU] 原始文本: {text}")

    # 去掉 @Bot 的提及 (飞书格式: @_user_1 或 @StockPredict Bot)
    text = re.sub(r"@[_\w]+\s*", "", text).strip()

    print(f"[FEISHU] 清理后文本: {text}")

    try:
        task = parse_message(text)
    except Exception as e:
        logger.error(f"parse_message failed: {e}", exc_info=True)
        _feishu_reply(chat_id, f"⚠️ 消息解析失败: {e}\n请稍后重试或使用 /help 查看可用指令")
        return jsonify({"status": "parse_error", "error": str(e)}), 200

    if not task:
        _feishu_reply(chat_id, f"无法识别指令。支持: /review /daily /analyze /pm /trader /scan")
        return jsonify({"status": "unknown_command"}), 200

    task.source = "feishu"
    task.sender = sender_id
    task.group_name = chat_id

    threading.Thread(target=_run_and_reply_feishu, args=(task, chat_id), daemon=True).start()

    return jsonify({"status": "accepted", "task_id": task.task_id}), 200


def _run_and_reply_feishu(task: AgentTask, chat_id: str):
    """执行任务并通过飞书回复结果"""
    execute_task(task)
    reply = format_reply(task)
    _feishu_reply(chat_id, reply)


def _feishu_reply(chat_id: str, message: str):
    """通过飞书 API 回复群消息"""
    cfg = _load_feishu_config()
    if not cfg.get("app_id") or not cfg.get("app_secret"):
        logger.warning("飞书 Bot 配置缺失")
        return

    token = _get_feishu_token(cfg["app_id"], cfg["app_secret"])
    if not token:
        logger.error("获取飞书 tenant_access_token 失败")
        return

    import requests
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    payload = {
        "receive_id": chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": message}),
    }

    try:
        requests.post(url, headers=headers, json=payload, params={"receive_id_type": "chat_id"}, timeout=10)
    except Exception as e:
        logger.error(f"飞书回复失败: {e}")


def _get_feishu_token(app_id: str, app_secret: str) -> str:
    """获取飞书 tenant_access_token"""
    import requests
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    payload = {"app_id": app_id, "app_secret": app_secret}
    try:
        resp = requests.post(url, json=payload, timeout=5)
        return resp.json().get("tenant_access_token", "")
    except Exception:
        return ""


def _load_feishu_config() -> dict:
    """加载飞书 Bot 配置"""
    from src.utils.config import load_config
    cfg = load_config()
    return {
        "app_id": cfg.get("feishu_app_id", ""),
        "app_secret": cfg.get("feishu_app_secret", ""),
        "verify_token": cfg.get("feishu_verify_token", ""),
        "encrypt_key": cfg.get("feishu_encrypt_key", ""),
    }


# ─── 通用 API ─────────────────────────────────────────────
@app.route("/api/tasks", methods=["GET"])
def list_tasks():
    """查看最近任务列表"""
    from .dispatcher import load_recent_tasks
    tasks = load_recent_tasks()
    return jsonify(tasks)


@app.route("/api/dispatch", methods=["POST"])
def manual_dispatch():
    """手动派发任务 (可用于测试)"""
    data = request.json or {}
    message = data.get("message", "")
    source = data.get("source", "api")

    task = parse_message(message)
    if not task:
        return jsonify({"error": "无法识别指令"}), 400

    task.source = source
    execute_task(task)
    reply = format_reply(task)
    return jsonify({"task_id": task.task_id, "agent": task.agent, "result": reply})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})


# ─── 启动 ─────────────────────────────────────────────────
def run_server(port: int = 8080, host: str = "0.0.0.0"):
    """启动 Bot webhook server"""
    feishu_cfg = _load_feishu_config()
    if feishu_cfg.get("verify_token"):
        global FEISHU_VERIFY_TOKEN
        FEISHU_VERIFY_TOKEN = feishu_cfg["verify_token"]
    if feishu_cfg.get("encrypt_key"):
        global FEISHU_ENCRYPT_KEY
        FEISHU_ENCRYPT_KEY = feishu_cfg["encrypt_key"]

    logger.info(f"Bot server starting on {host}:{port}")
    logger.info("Endpoints:")
    logger.info("  POST /pushplus/callback  — PushPlus 微信回调")
    logger.info("  POST /feishu/event       — 飞书事件回调")
    logger.info("  POST /api/dispatch       — 手动派发任务")
    logger.info("  GET  /api/tasks          — 查看任务列表")
    logger.info("  GET  /health             — 健康检查")

    app.run(host=host, port=port, debug=False, threaded=True)
