"""registry.py — 厂商/模型注册表 (models.dev 驱动).

数据解析顺序(照抄 Hermes 策略):
  1. 内存缓存(新鲜) 或 磁盘缓存(陈旧也先用,后台刷新)
  2. 磁盘缓存 models_dev_cache.json (含 ETag, 4h TTL, ETag 条件 GET)
  3. 网络(仅无缓存时阻塞拉取; 刷新失败退避5分钟)

私有覆盖层(内置, 优先于 models.dev):
  - zhipu-coding: 智谱 coding 套餐端点(用户实际持有的)
  - dashscope-cn: 百炼国内端点(models.dev 的 alibaba 挂的是国际端点)
  - alibaba-token: 用户私有 token-plan 端点(models.dev 没有)

所有厂商统一吐出一个扁平结构:
  {pid: {label, base_url, api_key_env_hint, models: {mid: {label, reasoning, ...}}}}
"""
import json
import logging
import threading
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CACHE_PATH = BASE_DIR / "models_dev_cache.json"
MODELS_DEV_URL = "https://models.dev/api.json"
CACHE_TTL = 4 * 3600
RETRY_DELAY = 300

_lock = threading.Lock()
_cache: dict | None = None          # models.dev 原始数据
_cache_time: float = 0
_etag: str = ""
_retry_after: float = 0

# ---------------------------------------------------------------- 私有覆盖
# key = models.dev provider id (或自定义 id); 优先于 registry 数据
OVERRIDES: dict[str, dict] = {
    "zhipu-coding": {
        "label": "智谱 GLM (Coding套餐)",
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
        "adapter": "zai",
        "models": {  # 2026-08-25 live 列表
            "glm-5.3": {"label": "GLM-5.3", "reasoning": True},
            "glm-5.2": {"label": "GLM-5.2", "reasoning": True},
            "glm-5.1": {"label": "GLM-5.1", "reasoning": True},
            "glm-5": {"label": "GLM-5", "reasoning": True},
            "glm-5-turbo": {"label": "GLM-5 Turbo", "reasoning": True},
            "glm-4.7": {"label": "GLM-4.7", "reasoning": True},
            "glm-4.6": {"label": "GLM-4.6", "reasoning": True},
            "glm-4.5": {"label": "GLM-4.5", "reasoning": True},
            "glm-4.5-air": {"label": "GLM-4.5 Air", "reasoning": True},
        },
    },
    "dashscope-cn": {
        "label": "阿里云百炼(国内)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "adapter": "dashscope",
        "models": {
            "qwen3.8-max": {"label": "Qwen3.8 Max", "reasoning": True},
            "qwen-plus": {"label": "Qwen Plus", "reasoning": False},
            "qwen-max": {"label": "Qwen Max", "reasoning": False},
            "qwen-turbo": {"label": "Qwen Turbo", "reasoning": False},
            "qwq-32b": {"label": "QwQ 32B", "reasoning": True},
        },
    },
    "alibaba-token": {
        "label": "阿里云Token套餐(私有)",
        "base_url": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "adapter": "dashscope",
        "models": {  # 2026-08-25 live 列表(GET /models 22个, 过滤非对话)
            "qwen3.7-max": {"label": "Qwen3.7 Max", "reasoning": True},
            "qwen3.7-plus": {"label": "Qwen3.7 Plus", "reasoning": True},
            "qwen3.6-plus": {"label": "Qwen3.6 Plus", "reasoning": True},
            "qwen3.6-flash": {"label": "Qwen3.6 Flash", "reasoning": True},
            "deepseek-v4-pro": {"label": "DeepSeek V4 Pro", "reasoning": True},
            "deepseek-v4-flash": {"label": "DeepSeek V4 Flash", "reasoning": True},
            "deepseek-v3.2": {"label": "DeepSeek V3.2", "reasoning": False},
            "glm-5.2": {"label": "GLM-5.2", "reasoning": True},
            "glm-5.1": {"label": "GLM-5.1", "reasoning": True},
            "glm-5": {"label": "GLM-5", "reasoning": True},
            "kimi-k2.6": {"label": "Kimi K2.6", "reasoning": False},
            "MiniMax-M2.5": {"label": "MiniMax M2.5", "reasoning": True},
        },
    },
    "moonshot-cn": {
        "label": "月之暗面 Kimi(国内)",
        "base_url": "https://api.moonshot.cn/v1",
        "adapter": "moonshot",
        "models": {
            "kimi-k2-0905-preview": {"label": "Kimi K2", "reasoning": False},
            "moonshot-v1-128k": {"label": "Moonshot V1 128K", "reasoning": False},
        },
    },
}

# models.dev id -> 我们展示的厂商 id(筛选 + 重命名)
VENDOR_MAP: dict[str, str] = {
    "deepseek": "deepseek",
    "minimax": "minimax",
    "openrouter": "openrouter",
    "siliconflow": "siliconflow",
    "stepfun": "stepfun",
    "zhipuai": "zhipu",
    "moonshotai-cn": "moonshot-cn",
}

# 国内用户贴 key 提示(Hermes .env 同名变量)
ENV_HINT = {
    "zhipu-coding": "GLM_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "alibaba-token": "HERMES_CUSTOM_ALIBABA_TOKEN_PLAN_API_KEY",
}


# ---------------------------------------------------------------- 缓存
def _load_disk_cache() -> None:
    global _cache, _cache_time, _etag
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        _cache = raw.get("data")
        _etag = raw.get("etag", "")
        _cache_time = raw.get("fetched_at", 0)
    except Exception:  # noqa: BLE001
        pass


def _fetch_registry(force: bool = False) -> None:
    """网络拉取(带 ETag); 失败退避。线程安全。"""
    global _cache, _cache_time, _etag, _retry_after
    with _lock:
        if _retry_after > time.time() and not force:
            return
        headers = {"If-None-Match": _etag} if _etag else {}
        try:
            r = requests.get(MODELS_DEV_URL, headers=headers, timeout=20)
            if r.status_code == 304:
                _cache_time = time.time()
                _retry_after = 0
                return
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict) or not data:
                raise ValueError("empty registry")
            _cache = data
            _cache_time = time.time()
            _etag = r.headers.get("ETag", "")
            CACHE_PATH.write_text(
                json.dumps({"data": data, "etag": _etag, "fetched_at": _cache_time}),
                encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("models.dev fetch failed: %s", exc)
            _retry_after = time.time() + RETRY_DELAY


def _ensure_loaded() -> dict:
    global _cache
    if _cache is not None:
        # 后台刷新(不阻塞)
        if _cache_time + CACHE_TTL < time.time():
            threading.Thread(target=_fetch_registry, daemon=True).start()
        return _cache
    _load_disk_cache()
    if _cache is not None:
        threading.Thread(target=_fetch_registry, daemon=True).start()
        return _cache
    _fetch_registry(force=True)
    return _cache or {}


# ---------------------------------------------------------------- 公开接口
def get_vendors(include_registry: bool = True) -> dict:
    """全部厂商: 私有覆盖 + models.dev 筛选集。

    返回 {pid: {label, base_url, adapter, models: {mid: {label, reasoning}}}}
    """
    vendors: dict = {}
    # 1. 私有覆盖(优先)
    for pid, ov in OVERRIDES.items():
        vendors[pid] = {
            "label": ov["label"],
            "base_url": ov["base_url"],
            "adapter": ov["adapter"],
            "builtin": True,
            "models": dict(ov["models"]),
        }
    # 2. models.dev 筛选集
    if include_registry:
        reg = _ensure_loaded()
        for reg_id, pid in VENDOR_MAP.items():
            p = reg.get(reg_id)
            if not isinstance(p, dict):
                continue
            models = {}
            for mid, m in (p.get("models") or {}).items():
                if not isinstance(m, dict):
                    continue
                # 过滤: 只要对话模型(排除 tts/asr/embedding/rerank 等)
                name = (m.get("name") or mid).lower()
                if any(x in mid.lower() or x in name for x in
                       ("tts", "asr", "audio", "embed", "rerank", "vision-encoder")):
                    continue
                models[mid] = {
                    "label": m.get("name") or mid,
                    "reasoning": bool(m.get("reasoning")),
                }
            if not models:
                continue
            vendors[pid] = {
                "label": p.get("name") or pid,
                "base_url": p.get("api") or "",
                "adapter": pid,  # 方言层按厂商id匹配
                "builtin": False,
                "models": models,
            }
    return vendors


def get_vendor(pid: str, custom_vendors: dict | None = None) -> dict | None:
    """取单个厂商(含用户自定义供应商)。"""
    if custom_vendors and pid in custom_vendors:
        return custom_vendors[pid]
    return get_vendors().get(pid)
