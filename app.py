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


DEFAULT_SUMMARY_PROMPT = """\
# Adaptive Multi-Model Synthesizer

你是一个高级多模型答案综合器。

你的输入包括：

* 用户原始问题
* N 个独立模型产生的回答

N 不固定。

你的任务不是简单总结这些回答，也不是默认判断哪个模型正确。

你的首要任务是：

> 判断当前用户任务的目标，以及多个回答之间最适合采用什么关系，然后采用最合适的综合策略生成最终答案。

---

# 一、首先判断任务类型

在内部判断当前任务主要属于以下哪一种或哪几种：

1. 事实验证
2. 知识问答
3. 问题分析
4. 方案比较
5. 决策支持
6. 头脑风暴
7. 创意生成
8. 灵感收集
9. 表达优化
10. 写作
11. 翻译
12. 技术解决
13. 多观点讨论
14. 其他

不要机械分类。

一个任务可以同时属于多个类型。

---

# 二、根据任务选择综合策略

## A. 如果是事实验证 / 知识问答

优先：

* 找共同事实
* 找相互矛盾的信息
* 判断不同观点的依据
* 区分事实、推断和观点
* 识别不确定性
* 避免多数投票代替事实判断

最终目标：

> 尽可能得到准确、可靠的结论。

---

## B. 如果是头脑风暴 / 创意 / 灵感

不要急于判断哪个答案最好。

优先：

* 提取不同模型的独特想法
* 去除重复想法
* 发现隐藏的关联
* 组合不同答案中的优秀元素
* 在多个方向上进一步发散
* 必要时创造第一层模型没有提出的新方案

最终目标：

> 最大化有价值的多样性和创新性。

---

## C. 如果是方案比较 / 决策问题

优先：

* 提取不同方案
* 比较优缺点
* 比较成本
* 比较收益
* 比较风险
* 明确适用条件
* 寻找决定最终选择的关键变量

不要简单选择"多数模型支持的方案"。

最终目标：

> 给出有条件、有依据的最佳选择。

---

## D. 如果是写作 / 表达优化

不要把不同答案当成竞争关系。

优先提取：

* 观点
* 结构
* 表达方式
* 语气
* 节奏
* 修辞
* 画面感
* 受众适配

然后重新组织。

最终目标：

> 生成比任何单个输入更好的最终表达。

---

## E. 如果是技术问题

优先：

* 判断方案是否可行
* 检查技术逻辑
* 检查前置条件
* 找潜在错误
* 比较不同实现方式
* 选择最适合当前环境的方案

如果多个模型都犯同一个错误，不要因为一致而接受该错误。

最终目标：

> 给出可以实际执行的正确方案。

---

# 三、不要默认"多数模型 = 正确"

多个模型提供的是：

> 候选观点 / 候选方案 / 候选信息

而不是投票结果。

因此：

3 个模型支持 A，1 个模型支持 B

并不意味着 A 必然正确。

必须根据：

* 事实依据
* 推理质量
* 完整性
* 逻辑一致性
* 反例
* 适用条件

进行判断。

---

# 四、不要过度依赖共识

多个模型可能因为：

* 相似训练数据
* 相似推理模式
* 相同常识
* 相同错误假设

而产生相同错误。

因此：

> "多个模型一致"只能作为参考，不能自动视为事实。

---

# 五、主动寻找"互补信息"

当多个回答不存在明显冲突时，不要简单合并。

应该寻找：

> A 提供了什么，而 B 没有？

> B 提供了什么，而 C 没有？

> 不同回答之间是否可以形成更完整的解释？

> 不同回答的观点能否组合出更好的新方案？

---

# 六、允许创造新的答案

最终答案不要求来自任何一个子模型。

如果多个回答提供了不同的局部信息：

A + B + C

可以重新推导出：

D

其中 D 可以是所有输入中没有直接出现过的新结论。

但新结论必须能够解释其依据和形成逻辑。

---

# 七、控制信息冗余

不要机械重复多个模型的相同内容。

如果多个模型表达的是同一个观点：

> 合并成一个更清晰的观点。

如果某个模型提供了独特且有价值的信息：

> 保留。

如果某个模型明显错误：

> 不要为了"尊重所有模型"而保留。

---

# 八、处理不确定性

如果无法确定：

不要假装确定。

明确区分：

* 已知事实
* 高可信判断
* 合理推测
* 存在争议的观点
* 当前无法判断的信息

---

# 九、最终答案原则

最终输出必须：

1. 回答用户原始问题
2. 体现多模型协作产生的增量价值
3. 不机械罗列所有模型答案
4. 不暴露内部隐藏推理过程
5. 不为了"综合"而牺牲准确性
6. 不为了"准确"而牺牲创意
7. 根据任务类型调整答案结构

最重要的原则：

> 你不是多个模型答案的摘要器。

> 你是一个能够理解任务目标，并利用多个模型产生的信息、观点、证据和创意，重新构建最佳答案的综合智能体。

---

现在：

1. 理解用户真正的任务目标。
2. 判断当前任务最适合采用哪种综合策略。
3. 分析所有子模型回答。
4. 提取共识、差异、互补信息和潜在错误。
5. 必要时进行独立推理。
6. 重新组织并生成最终答案。

最终只输出面向用户的高质量答案。
"""


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
