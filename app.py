"""Moa — 多模型协作问答工具.

将一个问题分发给多个 LLM 并发处理,各子模型的回答流式可见,
最后由主模型归纳总结。所有 provider 均走 OpenAI 兼容协议。
"""
import json
import os
import queue
import threading
import time
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"

app = Flask(__name__, static_folder=None)


# ---------------------------------------------------------------- config
def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {"providers": {}, "main": None}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


PRESETS = {
    "zai": {
        "label": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4.6",
        "docs": "https://open.bigmodel.cn",
    },
    "dashscope": {
        "label": "阿里云百炼",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "docs": "https://bailian.console.aliyun.com",
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "docs": "https://platform.deepseek.com",
    },
    "moonshot": {
        "label": "月之暗面 Kimi",
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "kimi-k2-0905-preview",
        "docs": "https://platform.moonshot.cn",
    },
    "minimax": {
        "label": "MiniMax",
        "base_url": "https://api.minimax.chat/v1",
        "default_model": "MiniMax-M2",
        "docs": "https://platform.minimaxi.com",
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
        "docs": "https://openrouter.ai",
    },
    "siliconflow": {
        "label": "硅基流动 SiliconFlow",
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "Qwen/Qwen2.5-72B-Instruct",
        "docs": "https://siliconflow.cn",
    },
    "ollama": {
        "label": "Ollama 本地",
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen2.5:7b",
        "docs": "https://ollama.com",
    },
}


def clean_api_key(key: str | None) -> str:
    """清洗 API Key:去首尾空白、剔除误粘贴的 ✓/✗ 前缀及一切非 ASCII 字符。

    requests 的 header 只接受 latin-1,key 里混入非 ASCII 会导致
    'latin-1 codec can't encode character' 错误。
    """
    key = (key or "").strip()
    key = key.strip("\u2713\u2717\u2714\u2718 ").strip()
    return "".join(ch for ch in key if 32 <= ord(ch) < 127)


# ---------------------------------------------------------------- chat core
def stream_one(provider_cfg: dict, model: str, question: str,
               out: queue.Queue, tag: str) -> None:
    """调用单个模型,把事件推入 out 队列;失败后重试 2 次(指数退避)。"""
    url = provider_cfg["base_url"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {clean_api_key(provider_cfg['api_key'])}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": True,
    }
    
    max_attempts = 2
    for attempt in range(max_attempts):
        start = time.time()
        n_chars = 0
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 stream=True, timeout=(10, 300))
            resp.raise_for_status()
            # 强制 UTF-8 解码,避免响应头 charset 缺失时 requests 用 latin-1 导致中文乱码
            resp.encoding = "utf-8"
            out.put(dict(event="start", tag=tag))
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                    piece = delta.get("content") or ""
                except (json.JSONDecodeError, IndexError):
                    piece = ""
                if piece:
                    n_chars += len(piece)
                    out.put(dict(event="delta", tag=tag, text=piece))
            out.put(dict(event="done", tag=tag, elapsed=round(time.time() - start, 1),
                         chars=n_chars))
            return  # 成功,退出重试循环
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)
            body = getattr(getattr(exc, "response", None), "text", "")[:300]
            if body:
                detail += f" | {body}"
            
            if attempt < max_attempts - 1:
                # 重试前等待(短间隔)
                wait = 0.3 * (2 ** attempt)
                out.put(dict(event="retry", tag=tag, attempt=attempt + 1,
                             wait=wait, message=detail))
                time.sleep(wait)
            else:
                # 最后一次失败,返回错误
                out.put(dict(event="error", tag=tag, message=detail,
                             elapsed=round(time.time() - start, 1)))


DEFAULT_SUMMARY_PROMPT = (
    "你是主控模型。不要逐个点评或罗列这些回答,而是:\n"
    "1. 汲取各回答中有价值的视角、论据与洞见;\n"
    "2. 剔除错误、片面或重复的内容,识别各模型的分歧点并独立判断谁更可信;\n"
    "3. 在此基础上形成你自己完整、连贯的最终答案——它应当比任何单一子模型的回答"
    "都更全面、更准确,直接回应问题本身。\n"
    "输出格式:直接给出你的最终答案(可分点/分层组织),如有必要可在末尾用一小段简述"
    "你采纳与舍弃了哪些观点及原因。用中文回答。"
)


def synth_summary(main_cfg: dict, question: str, answers: dict,
                  custom_prompt: str | None = None) -> str:
    """主模型汇总(非流式,一次性返回)。answers: {tag: {'text':..,'ok':bool}}"""
    parts = []
    for tag, a in answers.items():
        if a["ok"]:
            parts.append(f"【{tag}的回答】\n{a['text']}")
        else:
            parts.append(f"【{tag}】该模型调用失败:{a['error']}")
    joined = "\n\n".join(parts)
    instruction = (custom_prompt or "").strip() or DEFAULT_SUMMARY_PROMPT
    prompt = (
        f"用户的问题:\n{question}\n\n以下是多个 AI 模型从各自角度给出的回答:\n\n"
        f"{joined}\n\n{instruction}"
    )
    url = main_cfg["base_url"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {clean_api_key(main_cfg['api_key'])}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": main_cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return ((data.get("choices") or [{}])[0].get("message", {}) or {}).get(
        "content", "(主模型返回为空)")


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(BASE_DIR / "static", "index.html")


@app.get("/api/config")
def get_config():
    cfg = load_config()
    # 不向前端泄露 key 明文,只给掩码
    safe = {}
    for pid, p in cfg.get("providers", {}).items():
        q = dict(p)
        k = q.get("api_key", "")
        q["has_key"] = bool(k)
        q["api_key"] = (k[:4] + "****" + k[-4:]) if len(k) > 8 else ("****" if k else "")
        safe[pid] = q
    preset_list = [
        dict(id=k, **v) for k, v in PRESETS.items()
    ]
    return jsonify(providers=safe, presets=preset_list, main=cfg.get("main"),
                   default_summary_prompt=DEFAULT_SUMMARY_PROMPT,
                   summary_prompt=cfg.get("summary_prompt", ""))


@app.post("/api/config")
def set_config():
    """前端提交完整 provider 配置;api_key 若是掩码则保留旧值。"""
    incoming = request.get_json(force=True)
    cfg = load_config()
    old = cfg.setdefault("providers", {})
    merged = {}
    for pid, p in incoming.get("providers", {}).items():
        np = dict(p)
        if "****" in np.get("api_key", "") and pid in old:
            np["api_key"] = old[pid].get("api_key", "")
        merged[pid] = np
    cfg["providers"] = merged
    cfg["main"] = incoming.get("main")
    if "summary_prompt" in incoming:
        cfg["summary_prompt"] = (incoming.get("summary_prompt") or "").strip()
    save_config(cfg)
    return jsonify(ok=True)


@app.post("/api/test")
def test_provider():
    """快速连通性测试:发一条极小的非流式请求,检查响应体里的 choices 字段。"""
    body = request.get_json(force=True)
    base_url = (body.get("base_url") or "").strip().rstrip("/")
    api_key = (body.get("api_key") or "").strip()
    model = (body.get("model") or "").strip()
    # 掩码 key → 用已存的真实 key
    if "****" in api_key:
        pid = body.get("pid")
        stored = load_config().get("providers", {}).get(pid, {}).get("api_key", "")
        api_key = stored
    api_key = clean_api_key(api_key)
    if not base_url or not model:
        return jsonify(ok=False, error="Base URL 和模型名不能为空")
    try:
        resp = requests.post(
            base_url + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 1, "stream": False},
            timeout=20,
        )
        if resp.status_code != 200:
            detail = resp.text[:200]
            return jsonify(ok=False, error=f"HTTP {resp.status_code}: {detail}")
        # 检查响应体里的 choices 字段
        try:
            data = resp.json()
            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                return jsonify(ok=False, error=f"响应格式错误: {str(data)[:200]}")
            return jsonify(ok=True)
        except json.JSONDecodeError as e:
            return jsonify(ok=False, error=f"JSON 解析失败: {str(e)}")
    except Exception as exc:  # noqa: BLE001
        return jsonify(ok=False, error=str(exc)[:200])


@app.post("/api/ask")
def ask():
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify(error="问题为空"), 400

    cfg = load_config()
    providers = cfg.get("providers", {})
    selected = [p for p in body.get("models", [])
                if p in providers and providers[p].get("api_key")]
    if not selected:
        return jsonify(error="没有可用的已配置模型"), 400

    main_pid = body.get("main") or cfg.get("main")
    main_cfg = providers.get(main_pid) if main_pid else None
    
    # 主模型不能同时作为子模型
    if main_pid and main_pid in selected:
        selected.remove(main_pid)
    
    if not selected:
        return jsonify(error="至少需要一个非主模型的子模型"), 400
    custom_prompt = (body.get("summary_prompt")
                     or cfg.get("summary_prompt") or "").strip() or None

    def generate():
        qout = queue.Queue()
        threads = []
        for tag in selected:
            pcfg = providers[tag]
            t = threading.Thread(
                target=stream_one,
                args=(pcfg, pcfg.get("model", ""), question, qout, tag),
                daemon=True,
            )
            t.start()
            threads.append(t)

        buf: dict[str, str] = {}   # tag -> 累积全文
        errors: dict[str, str] = {}
        finished: set[str] = set()
        summary_sent = False

        while True:
            try:
                ev = qout.get(timeout=600)
            except queue.Empty:
                yield ndjson(dict(event="fatal", message="等待超时"))
                break

            kind = ev["event"]
            if kind == "delta":
                buf[ev["tag"]] = buf.get(ev["tag"], "") + ev["text"]
            elif kind == "error":
                errors[ev["tag"]] = ev.get("message", "unknown")
                finished.add(ev["tag"])
            elif kind == "done":
                finished.add(ev["tag"])

            yield ndjson(ev)

            if finished == set(selected) and not summary_sent:
                summary_sent = True
                if main_cfg and main_cfg.get("api_key"):
                    yield ndjson(dict(event="summary_start"))
                    answers = {
                        tag: {
                            "ok": tag not in errors,
                            "text": buf.get(tag, ""),
                            "error": errors.get(tag, ""),
                        }
                        for tag in selected
                    }
                    try:
                        text = synth_summary(main_cfg, question, answers,
                                             custom_prompt=custom_prompt)
                        for ch in chunks(text):
                            yield ndjson(dict(event="summary_delta", text=ch))
                        yield ndjson(dict(event="summary_done"))
                    except Exception as exc:  # noqa: BLE001
                        yield ndjson(dict(
                            event="summary_error",
                            message=f"{exc} | "
                                    f"{getattr(getattr(exc, 'response', None), 'text', '')[:200]}",
                        ))
                else:
                    yield ndjson(dict(event="summary_skipped",
                                      message="未配置主模型,跳过汇总"))
                break

        yield ndjson(dict(event="all_done"))

    return Response(generate(), mimetype="application/x-ndjson")


def ndjson(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def chunks(text: str, size: int = 24):
    for i in range(0, len(text), size):
        yield text[i:i + size]


if __name__ == "__main__":
    os.makedirs(BASE_DIR / "static", exist_ok=True)
    app.run(host="127.0.0.1", port=7819, debug=False, threaded=True)
