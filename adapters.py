"""adapters.py — 思考参数方言层 + 三级连通性测试.

方言层只做一件事: (pid, model, thinking_on) -> payload 修改。
联网不走这里(应用侧统一搜索,见 searchhub.py)。
"""
import json

import requests

from registry import get_vendor


# ---------------------------------------------------------------- 方言函数
# effort 等级: "low" | "medium" | "high" (None=厂商默认)

def _zai_think(p, effort=None):
    p["thinking"] = {"type": "enabled"}
    if effort:
        # 智谱: 无 effort 字段,部分型号支持 max_thinking_tokens,通用起见忽略等级
        pass


def _ds_think(p, effort=None):
    p["enable_thinking"] = True
    if effort == "low":
        p["thinking_budget"] = 2048
    elif effort == "high":
        p["thinking_budget"] = 16384
    # medium/None: 厂商默认


def _ds_think_fl(p, effort=None):
    """dashscope 兼容层 thinking 对象(新规范)。"""
    p["enable_thinking"] = True
    budgets = {"low": 2048, "medium": 8192, "high": 16384}
    if effort and effort in budgets:
        p["thinking_budget"] = budgets[effort]


def _deepseek_think(p, effort=None):
    # deepseek-v4 系: reasoning_effort
    if effort:
        p["reasoning_effort"] = effort


def _noop(p, effort=None):
    pass


# pid -> thinking 参数方言 (联网能力全部移除, 应用侧搜索替代)
THINK_DIALECTS: dict[str, object] = {
    "zai": _zai_think,                       # thinking 对象 (实测)
    "zhipu": _zai_think,
    "zhipu-coding": _zai_think,
    "dashscope": _ds_think_fl,               # enable_thinking + thinking_budget (实测)
    "dashscope-cn": _ds_think_fl,
    "alibaba-token": _ds_think_fl,
    "deepseek": _deepseek_think,             # reasoning_effort (v4系实测口径)
    "moonshot": _noop,
    "moonshot-cn": _noop,
    "minimax": _noop,
    "openrouter": _noop,                     # 模型名区分
    "siliconflow": _noop,
    "stepfun": _noop,
    "generic": _noop,                        # 自定义供应商: 不猜方言
}


def build_payload(vendor: dict, model: str, messages: list,
                  thinking: bool = False, effort: str | None = None,
                  stream: bool = True, max_tokens: int | None = None) -> dict:
    """唯一 payload 构建入口。

    vendor: {base_url, adapter/pid, ...}; model: 模型id;
    thinking: 是否开思考; effort: 思考等级 low/medium/high(None=默认)。
    """
    payload = {"model": model, "messages": messages, "stream": stream}
    dialect = THINK_DIALECTS.get(vendor.get("adapter") or vendor.get("pid") or "generic",
                                 _noop)
    if thinking:
        dialect(payload, effort)
    if max_tokens:
        payload["max_tokens"] = max_tokens
    return payload


# ---------------------------------------------------------------- 三级连通测试
def probe(base_url: str, api_key: str, model: str, adapter: str,
          thinking: bool, effort: str | None = None, timeout: int = 15) -> dict:
    """三级探测, 返回 {ok, level, detail, models?}。

    level 1: GET /models        — 端点+key (零成本)
    level 2: 最小真实对话        — 模型名+配额+参数 (≈10 token)
    level 3: 思考流验证          — reasoning_content 真的在流里 (与2合并)
    """
    base = (base_url or "").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    result = {"ok": False, "level": 0, "detail": ""}

    # ---- level 1: /models
    models_list = []
    try:
        r1 = requests.get(f"{base}/models", headers=headers, timeout=timeout)
        if r1.status_code == 401 or r1.status_code == 403:
            result.update(level=1, detail=f"key 无效或无权限 (HTTP {r1.status_code}): {r1.text[:120]}")
            return result
        if r1.status_code == 404:
            # 有的端点不开放 /models, 不算致命, 继续二级
            models_list = []
        elif r1.status_code == 200:
            try:
                data = r1.json()
                models_list = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)]
            except json.JSONDecodeError:
                result.update(level=1, detail="端点返回了非 JSON(疑似 URL 错误)")
                return result
        else:
            result.update(level=1, detail=f"端点异常 HTTP {r1.status_code}: {r1.text[:120]}")
            return result
    except requests.RequestException as exc:
        result.update(level=1, detail=f"连不上端点: {exc}")
        return result

    # ---- level 2+3: 最小真实对话(与正式调用同路径)
    if not model:
        result.update(ok=True, level=1, detail="端点与 key 有效(未测模型)",
                      models=models_list)
        return result

    vendor = {"base_url": base, "adapter": adapter, "pid": adapter}
    payload = build_payload(vendor, model,
                            [{"role": "user", "content": "hi"}],
                            thinking=thinking, effort=effort, stream=True,
                            max_tokens=16)
    try:
        r2 = requests.post(f"{base}/chat/completions", headers=headers,
                           json=payload, stream=True, timeout=timeout)
        if r2.status_code != 200:
            txt = r2.text[:150]
            if "insufficient_quota" in txt or "exhausted" in txt or "余额不足" in txt:
                return {"ok": False, "level": 2,
                        "detail": "💰 配额已耗尽(账号额度用完,非配置问题)——充值/等额度刷新后即用。端点与 key 均有效。"}
            result.update(level=2, detail=f"模型调用失败 HTTP {r2.status_code}: {txt}")
            return result
        # 读流: 检查 content 和 reasoning_content
        saw_content = False
        saw_reasoning = False
        for raw in r2.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                if delta.get("content"):
                    saw_content = True
                if delta.get("reasoning_content") or delta.get("thinking"):
                    saw_reasoning = True
            except (json.JSONDecodeError, IndexError):
                continue
        if not saw_content and not saw_reasoning:
            result.update(level=2, detail="端点200但流内无内容(模型名错误?)")
            return result
        if thinking and not saw_reasoning:
            result.update(level=3, detail="调用成功,但思考参数未生效(该端点忽略思考参数)")
            return result
        detail = "key 有效,模型响应"
        if thinking:
            detail += ",思考流确认 ✓"
        result.update(ok=True, level=3, detail=detail, models=models_list)
        return result
    except requests.RequestException as exc:
        result.update(level=2, detail=f"模型调用异常: {exc}")
        return result


def probe_vendor(pid: str, api_key: str, model: str, thinking: bool,
                 custom_vendors: dict | None = None) -> dict:
    """按厂商id探测(registry 查端点+方言)。"""
    v = get_vendor(pid, custom_vendors)
    if not v:
        return {"ok": False, "level": 0, "detail": f"未知厂商 {pid}"}
    return probe(v["base_url"], api_key, model, v.get("adapter") or pid, thinking)
