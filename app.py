"""Moa v3 — 多模型协作问答 (models.dev 驱动).

架构:
  registry.py   厂商/模型数据 (models.dev + 私有覆盖 + 自定义供应商)
  adapters.py   思考方言层 + 三级连通测试
  searchhub.py  应用侧统一搜索 (anysearch 优先, ddgs 兜底)
  app.py        Flask 路由 + NDJSON 流式管线

事件协议: start/thinking/delta/retry/error/done
          → summary_start/summary_delta*/summary_done/summary_error/all_done
汇总为真流式(主模型 stream=True 逐 token 转发)。
"""
import json
import os
import queue
import threading
import time
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_from_directory

import searchhub
from adapters import build_payload, probe_vendor
from registry import get_vendor, get_vendors

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
app = Flask(__name__, static_folder=None)


# ================================================================ config
def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {"keys": {}, "custom_vendors": {}, "main": None, "search": {},
            "summary_prompt": ""}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                           encoding="utf-8")


def clean_api_key(key: str | None) -> str:
    key = (key or "").strip()
    key = key.strip("\u2713\u2717\u2714\u2718 ").strip()
    return "".join(ch for ch in key if 32 <= ord(ch) < 127)


DEFAULT_SUMMARY_PROMPT = (
    "# Role\n"
    "你是一个高级认知处理中枢。你的输入是三个AI助理对同一问题的回复。你的任务不是盲目汇总,"
    "而是先对这三份回复的\"同质化程度\"和\"内容属性\"进行分类,然后自动切换到最合适的处理模式。\n\n"
    "# 第一步:场景预判(请先默默在心中执行以下分类)\n"
    "在阅读回复前,先判断这三份回答属于哪种类型:\n\n"
    "高度共识型(结论90%相似,仅表述不同) -> 切换到 模式A:精华萃取师\n\n"
    "百花齐放型(角度完全不同,但都有道理) -> 切换到 模式B:创意策展人\n\n"
    "事实冲突型(数据、代码、日期、逻辑存在硬性矛盾) -> 切换到 模式C:终极仲裁员\n\n"
    "信息残缺型(每份回答都只解决了问题的一部分) -> 切换到 模式D:拼图缝合师\n\n"
    "# 第二步:根据预判模式,执行对应输出规则\n\n"
    "A 模式 A:精华萃取师(当回答大同小异时)\n"
    "目标:浓缩而非扩充。\n"
    "规则:剔除所有重复的铺垫语、客套话。提取三份回答中最高频出现的核心方法论。\n"
    "输出格式:用一段极简干货直击要点,字数不超过任意一份子回答的 60%。"
    "末尾附上\"三份回答共识度 95%,以上为去重后的最简方案。\"\n\n"
    "B 模式 B:创意策展人(当角度百花齐放时)\n"
    "目标:保留多样性,提供可选择性。\n"
    "规则:不要强行融合(强行融合会变四不像)。将三份回答归类为\"路径1\"、\"路径2\"、\"路径3\"。\n"
    "输出格式:列出三份回答各自的\"最独特亮点\",并附上\"该路径最适合的人群/场景\""
    "(例如:路径A适合预算充足者,路径B适合急性子)。让用户根据自身情况自助选择,你只做导购,不做决策。\n\n"
    "C 模式 C:终极仲裁员(当存在硬性冲突时)\n"
    "目标:基于逻辑和常识去伪存真。\n"
    "规则:严格调用你内置的通用知识库进行逻辑检验。不去看谁说的\"权威\",只看谁说的\"合理\"。\n"
    "输出格式:明确指出矛盾点在哪里,并给出你的判断依据。如果无法判断,必须追问用户提供额外上下文,严禁瞎编圆场。\n\n"
    "D 模式 D:拼图缝合师(当信息残缺互补时)\n"
    "目标:按逻辑时序重组。\n"
    "规则:将三份答案拆解为碎片化步骤,按\"时间顺序\"、\"重要程度\"或\"输入->处理->输出\"的逻辑链条重新排列。\n"
    "输出格式:输出一份无缝衔接的 SOP(标准作业流程),确保每一步之间逻辑连贯,不留空白。\n\n"
    "# 第三步:强制启动语\n"
    "在输出最终回答的最开头,请先用括号标注你本次选用的模式(例如:【本次采用模式B:创意策展人】),然后再输出正文。"
)


# ================================================================ 流式调用
def stream_one(vendor: dict, api_key: str, model: str, question: str,
               thinking: bool, effort: str | None, out: queue.Queue, tag: str) -> None:
    """单个子模型: 流式调用,事件推队列;按错误类别分流重试。"""
    url = vendor["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {clean_api_key(api_key)}",
               "Content-Type": "application/json"}
    payload = build_payload(vendor, model,
                            [{"role": "user", "content": question}],
                            thinking=thinking, effort=effort, stream=True)

    max_attempts = 3
    for attempt in range(max_attempts):
        start = time.time()
        n_chars = 0
        t_chars = 0
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 stream=True, timeout=(10, 300))
            resp.raise_for_status()
            resp.encoding = "utf-8"
            out.put(dict(event="start", tag=tag))
            for raw in resp.iter_lines():
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
                    th = delta.get("reasoning_content") or delta.get("thinking") or ""
                    if th:
                        t_chars += len(th)
                        out.put(dict(event="thinking", tag=tag, text=th))
                    piece = delta.get("content") or ""
                except (json.JSONDecodeError, IndexError):
                    piece = ""
                if piece:
                    n_chars += len(piece)
                    out.put(dict(event="delta", tag=tag, text=piece))
            out.put(dict(event="done", tag=tag,
                         elapsed=round(time.time() - start, 1),
                         chars=n_chars, thinking_chars=t_chars))
            return
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)
            status = getattr(getattr(exc, "response", None), "status_code", None)
            body = getattr(getattr(exc, "response", None), "text", "")[:300]
            if body:
                detail += f" | {body}"
            # 配额/余额类错误: 账号层面,重试无意义,直接人话报错
            quota_hit = any(x in detail for x in (
                "insufficient_quota", "quota has been exhausted",
                "quota exhausted", "余额不足", "无可用资源包", "Arrearage"))
            if quota_hit:
                out.put(dict(event="error", tag=tag,
                             message="💰 配额已耗尽(账号层面,非配置问题)——该厂商额度用完,"
                                     "请到控制台充值或等额度刷新。其他厂商不受影响。",
                             elapsed=round(time.time() - start, 1)))
                return
            retryable = status is None or status in (408, 429) or (status or 0) >= 500
            if retryable and attempt < max_attempts - 1:
                wait = min(2 ** attempt, 8)
                out.put(dict(event="retry", tag=tag, attempt=attempt + 1,
                             wait=wait, message=detail))
                time.sleep(wait)
            else:
                out.put(dict(event="error", tag=tag, message=detail,
                             elapsed=round(time.time() - start, 1)))


def stream_summary(vendor: dict, api_key: str, model: str, prompt: str,
                   out: queue.Queue) -> None:
    """主模型汇总: 真流式,逐 token 推 summary_delta。"""
    url = vendor["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {clean_api_key(api_key)}",
               "Content-Type": "application/json"}
    payload = build_payload(vendor, model,
                            [{"role": "user", "content": prompt}],
                            thinking=False, stream=True)
    try:
        resp = requests.post(url, headers=headers, json=payload,
                             stream=True, timeout=(10, 600))
        resp.raise_for_status()
        resp.encoding = "utf-8"
        out.put(dict(event="summary_start"))
        for raw in resp.iter_lines():
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
                th = delta.get("reasoning_content") or ""
                if th:
                    out.put(dict(event="summary_thinking", text=th))
                piece = delta.get("content") or ""
            except (json.JSONDecodeError, IndexError):
                piece = ""
            if piece:
                out.put(dict(event="summary_delta", text=piece))
        out.put(dict(event="summary_done"))
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
        body = getattr(getattr(exc, "response", None), "text", "")[:300]
        if body:
            detail += f" | {body}"
        out.put(dict(event="summary_error", message=detail))


# ================================================================ routes
@app.get("/")
def index():
    return send_from_directory(BASE_DIR / "static", "index.html")


@app.get("/static/<path:fname>")
def static_files(fname):
    return send_from_directory(BASE_DIR / "static", fname)


@app.get("/api/vendors")
def api_vendors():
    """厂商全集(registry + 用户自定义), key 已掩码。"""
    cfg = load_config()
    keys = cfg.get("keys", {})
    vendors = get_vendors()
    # 自定义供应商
    for cid, cv in (cfg.get("custom_vendors") or {}).items():
        vendors[cid] = {
            "label": cv.get("label") or cid,
            "base_url": cv.get("base_url", ""),
            "adapter": "generic",
            "builtin": False,
            "custom": True,
            "models": cv.get("models") or {},
        }
    safe = {}
    for pid, v in vendors.items():
        q = dict(v)
        k = keys.get(pid, "")
        q["has_key"] = bool(k)
        q["key_mask"] = (k[:4] + "****" + k[-4:]) if len(k) > 8 else ("****" if k else "")
        q.pop("api_key", None)
        safe[pid] = q
    return jsonify(vendors=safe, main=cfg.get("main"),
                   default_summary_prompt=DEFAULT_SUMMARY_PROMPT,
                   summary_prompt=cfg.get("summary_prompt", ""),
                   search=cfg.get("search", {}))


@app.post("/api/keys")
def api_keys():
    """保存/清除某厂商 key。body: {pid, api_key} (空串=清除)。"""
    body = request.get_json(force=True)
    cfg = load_config()
    pid = body.get("pid") or ""
    key = clean_api_key(body.get("api_key") or "")
    if not pid:
        return jsonify(ok=False, error="pid 必填")
    if key:
        cfg.setdefault("keys", {})[pid] = key
    else:
        cfg.setdefault("keys", {}).pop(pid, None)
    save_config(cfg)
    return jsonify(ok=True)


@app.post("/api/custom_vendors")
def api_custom_vendors():
    """增删自定义供应商。body: {action: add|remove, vendor: {...}}"""
    body = request.get_json(force=True)
    cfg = load_config()
    action = body.get("action")
    v = body.get("vendor") or {}
    if action == "add":
        cid = (v.get("id") or "").strip() or f"custom_{int(time.time())}"
        cfg.setdefault("custom_vendors", {})[cid] = {
            "label": v.get("label") or cid,
            "base_url": (v.get("base_url") or "").strip().rstrip("/"),
            "models": v.get("models") or {},
        }
        save_config(cfg)
        return jsonify(ok=True, id=cid)
    if action == "remove":
        cfg.setdefault("custom_vendors", {}).pop(v.get("id"), None)
        save_config(cfg)
        return jsonify(ok=True)
    return jsonify(ok=False, error="未知 action")


@app.post("/api/fetch_models")
def api_fetch_models():
    """拉取自定义供应商的 /models 列表。body: {base_url, api_key}"""
    body = request.get_json(force=True)
    base = (body.get("base_url") or "").strip().rstrip("/")
    key = clean_api_key(body.get("api_key") or "")
    if not base:
        return jsonify(ok=False, error="Base URL 必填")
    if "****" in (body.get("api_key") or ""):
        return jsonify(ok=False, error="请粘贴完整 key(掩码不可用)")
    try:
        r = requests.get(f"{base}/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=15)
        if r.status_code != 200:
            return jsonify(ok=False, error=f"HTTP {r.status_code}: {r.text[:150]}")
        data = r.json()
        ids = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)]
        return jsonify(ok=True, models=ids)
    except Exception as exc:  # noqa: BLE001
        return jsonify(ok=False, error=str(exc)[:200])


@app.post("/api/probe")
def api_probe():
    """三级连通测试。body: {pid, api_key?, model, thinking, effort?}"""
    body = request.get_json(force=True)
    pid = body.get("pid") or ""
    model = body.get("model") or ""
    thinking = bool(body.get("thinking"))
    effort = body.get("effort")
    cfg = load_config()
    key = body.get("api_key")
    if not key or "****" in key:
        key = cfg.get("keys", {}).get(pid, "")
    key = clean_api_key(key or "")
    v = get_vendor(pid, cfg.get("custom_vendors"))
    if not v:
        return jsonify(ok=False, level=0, detail=f"未知厂商 {pid}")
    from adapters import probe as _probe
    payload_adapter = v.get("adapter") or pid
    result = _probe(v["base_url"], key, model, payload_adapter, thinking,
                    effort=effort)
    return jsonify(result)


@app.post("/api/vendor_models")
def api_vendor_models():
    """已贴key厂商: 实时拉 live 模型列表并与预置合并。body: {pid}"""
    body = request.get_json(force=True)
    pid = body.get("pid") or ""
    cfg = load_config()
    key = clean_api_key(cfg.get("keys", {}).get(pid, ""))
    v = get_vendor(pid, cfg.get("custom_vendors"))
    if not v:
        return jsonify(ok=False, error=f"未知厂商 {pid}")
    if not key:
        return jsonify(ok=False, error="该厂商未配置 key")
    base = v["base_url"].rstrip("/")
    preset = v.get("models", {})
    try:
        r = requests.get(f"{base}/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=15)
        if r.status_code != 200:
            # live 拉取失败: 退回预置(不算致命)
            return jsonify(ok=True, models=preset, live=False,
                           error=f"live 拉取失败(HTTP {r.status_code}),显示预置列表")
        ids = [m.get("id") for m in r.json().get("data", []) if isinstance(m, dict) and m.get("id")]
        # 合并: live 优先(排序按 live 顺序),预置补漏
        merged = {}
        preset_adapter = v.get("adapter") or pid
        from adapters import THINK_DIALECTS
        for mid in ids:
            if mid in preset:
                merged[mid] = preset[mid]
            else:
                # 新模型: 按预置同家族推断 reasoning(保守:名字含思考关键词)
                reasoning = bool(any(x in mid.lower() for x in
                                     ("glm-4.6", "glm-4.7", "glm-5", "qwen3", "r1", "reasoner", "v4", "thinking", "qwq")))
                merged[mid] = {"label": mid, "reasoning": reasoning}
        for mid, m in preset.items():
            if mid not in merged:
                merged[mid] = m
        return jsonify(ok=True, models=merged, live=True, live_count=len(ids))
    except Exception as exc:  # noqa: BLE001
        return jsonify(ok=True, models=preset, live=False, error=f"live 拉取异常: {str(exc)[:100]}")


@app.post("/api/search_test")
def api_search_test():
    """测试搜索源。body: {question}"""
    body = request.get_json(force=True)
    q = (body.get("question") or "").strip() or "测试:今日新闻"
    cfg = load_config()
    refs = searchhub.search(q, cfg.get("search", {}))
    return jsonify(ok=bool(refs), chars=len(refs), preview=refs[:300])


@app.post("/api/save")
def api_save():
    """保存杂项: main / summary_prompt / search 配置。"""
    body = request.get_json(force=True)
    cfg = load_config()
    if "main" in body:
        cfg["main"] = body["main"]
    if "summary_prompt" in body:
        cfg["summary_prompt"] = (body["summary_prompt"] or "").strip()
    if "search" in body:
        s = body["search"] or {}
        clean = {"anysearch_key": clean_api_key(s.get("anysearch_key") or "")}
        # 掩码保留旧值
        if "****" in clean["anysearch_key"]:
            clean["anysearch_key"] = cfg.get("search", {}).get("anysearch_key", "")
        cfg["search"] = clean
    save_config(cfg)
    return jsonify(ok=True)


@app.post("/api/ask")
def api_ask():
    """主流程: 并发子模型 + 可选搜索 + 真流式汇总。"""
    body = request.get_json(force=True)
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify(error="问题为空"), 400

    cfg = load_config()
    keys = cfg.get("keys", {})
    vendors = get_vendors()
    for cid, cv in (cfg.get("custom_vendors") or {}).items():
        vendors[cid] = {"label": cv.get("label") or cid, "base_url": cv.get("base_url", ""),
                        "adapter": "generic", "models": cv.get("models") or {}}

    # selected: [{pid, model, thinking, effort}]
    selected = []
    for item in body.get("models", []):
        pid = item.get("pid")
        model = item.get("model")
        if pid in vendors and model and keys.get(pid):
            selected.append({"pid": pid, "model": model,
                             "thinking": bool(item.get("thinking")),
                             "effort": item.get("effort")})
    if not selected:
        return jsonify(error="没有可用的已配置模型(缺 key 或未选)"), 400

    # main 传 [pid, model] 或 null; 未传时用配置默认
    main_sel = body.get("main")
    if isinstance(main_sel, list) and len(main_sel) == 2:
        main_pid, main_model = main_sel
        if main_pid not in keys:
            main_pid = None
    else:
        m = cfg.get("main")
        if isinstance(m, list) and len(m) == 2 and m[0] in keys:
            main_pid, main_model = m
        else:
            main_pid = None

    # 主模型不能同时是子模型
    if main_pid:
        selected = [s for s in selected
                    if not (s["pid"] == main_pid and s["model"] == main_model)]
    if not selected:
        return jsonify(error="至少需要一个非主模型的子模型"), 400

    use_search = bool(body.get("search"))
    custom_prompt = (body.get("summary_prompt") or cfg.get("summary_prompt")
                     or "").strip() or None

    def generate():
        # 1. 搜索(联网开时)
        refs = ""
        if use_search:
            yield nd(dict(event="search_start"))
            refs = searchhub.search(question, cfg.get("search", {}))
            yield nd(dict(event="search_done", chars=len(refs)))

        # 2. 并发子模型
        full_q = (question + "\n\n" + refs) if refs else question
        qout: queue.Queue = queue.Queue()
        tags = []
        for s in selected:
            tag = f'{s["pid"]}/{s["model"]}'
            tags.append(tag)
            threading.Thread(
                target=stream_one,
                args=(vendors[s["pid"]], keys[s["pid"]], s["model"],
                      full_q, s["thinking"], s.get("effort"), qout, tag),
                daemon=True,
            ).start()

        buf: dict[str, str] = {}
        errors: dict[str, str] = {}
        finished: set[str] = set()
        summary_sent = False

        while True:
            try:
                ev = qout.get(timeout=600)
            except queue.Empty:
                yield nd(dict(event="fatal", message="等待超时"))
                break
            kind = ev["event"]
            if kind == "delta":
                buf[ev["tag"]] = buf.get(ev["tag"], "") + ev["text"]
            elif kind == "error":
                errors[ev["tag"]] = ev.get("message", "unknown")
                finished.add(ev["tag"])
            elif kind == "done":
                finished.add(ev["tag"])
            if kind != "summary_start":
                yield nd(ev)

            if finished == set(tags) and not summary_sent:
                summary_sent = True
                if main_pid:
                    parts = []
                    for tag in tags:
                        if tag in errors:
                            parts.append(f"【{tag}】调用失败:{errors[tag]}")
                        else:
                            parts.append(f"【{tag}的回答】\n{buf.get(tag, '')}")
                    instruction = custom_prompt or DEFAULT_SUMMARY_PROMPT
                    # 提示词里的"三个/三份"动态适配实际子模型数(用户提示词硬编码了三)
                    n = len(tags)
                    if n != 3:
                        instruction = instruction.replace("三个", f"{n}个").replace("三份", f"{n}份")
                    prompt = (
                        f"用户的问题:\n{question}\n\n"
                        f"以下是多个 AI 模型从各自角度给出的回答:\n\n"
                        f"\n\n".join(parts) + f"\n\n{instruction}"
                    )
                    # 汇总线程
                    sout: queue.Queue = queue.Queue()
                    threading.Thread(
                        target=stream_summary,
                        args=(vendors[main_pid], keys[main_pid], main_model,
                              prompt, sout),
                        daemon=True,
                    ).start()
                    while True:
                        try:
                            sev = sout.get(timeout=600)
                        except queue.Empty:
                            yield nd(dict(event="summary_error", message="汇总超时"))
                            break
                        yield nd(sev)
                        if sev["event"] in ("summary_done", "summary_error"):
                            break
                else:
                    yield nd(dict(event="summary_skipped",
                                  message="未指定主模型,跳过汇总"))
                break

        yield nd(dict(event="all_done"))

    return Response(generate(), mimetype="application/x-ndjson")


def nd(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


if __name__ == "__main__":
    os.makedirs(BASE_DIR / "static", exist_ok=True)
    app.run(host="127.0.0.1", port=7819, debug=False, threaded=True)
