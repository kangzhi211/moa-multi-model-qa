"""searchhub.py — 应用侧统一搜索层.

所有子模型基于同一份搜索材料作答(对比变量干净)。
源:
  - anysearch: HTTP MCP (贴 key 启用, 质量高, 中文友好) — 优先
  - ddgs: 免费无 key, 兜底

对外只暴露: search(question, cfg) -> str(拼接好的参考资料,空串=没搜到)
"""
import json
import logging

import requests

logger = logging.getLogger(__name__)

ANYSEARCH_URL = "https://api.anysearch.com/mcp"


# ---------------------------------------------------------------- anysearch (HTTP MCP)
def _search_anysearch(question: str, api_key: str, max_results: int = 6) -> str:
    """直连 anysearch HTTP MCP, 返回 markdown 结果文本(已是好格式,直接用)。"""
    try:
        r = requests.post(
            ANYSEARCH_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Anysearch-Client": "mcp/1.0.0",
                "Accept": "application/json, text/event-stream",
            },
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": "search",
                             "arguments": {"query": question, "max_results": max_results}}},
            timeout=25,
        )
        if r.status_code != 200:
            logger.warning("anysearch HTTP %s", r.status_code)
            return ""
        data = r.json()
        if "error" in data:
            logger.warning("anysearch error: %s", data["error"])
            return ""
        content = ((data.get("result") or {}).get("content") or [])
        texts = [c.get("text", "") for c in content if isinstance(c, dict)]
        return texts[0] if texts else ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("anysearch failed: %s", exc)
        return ""


# ---------------------------------------------------------------- ddgs (兜底)
def _search_ddgs(question: str, max_results: int = 6) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return ""
    try:
        with DDGS() as d:
            results = d.text(question, max_results=max_results)
        if not results:
            return ""
        lines = []
        for i, rr in enumerate(results, 1):
            lines.append(f"{i}. {rr.get('title', '')}\n   {rr.get('href') or rr.get('url', '')}\n   {rr.get('body', '')[:200]}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ddgs failed: %s", exc)
        return ""


# ---------------------------------------------------------------- 统一入口
def search(question: str, cfg: dict | None = None) -> str:
    """搜索并拼接为 prompt 参考资料。失败返回空串(不阻塞提问)。"""
    cfg = cfg or {}
    header = f"[联网搜索资料 · 与问题「{question}」相关]\n"
    footer = "\n[以上为搜索参考资料,回答时可作为依据,不确定的可声明]"

    if cfg.get("anysearch_key"):
        md = _search_anysearch(question, cfg["anysearch_key"])
        if md:
            return header + md + footer
        logger.info("anysearch 未返回,降级 ddgs")

    dd = _search_ddgs(question)
    if dd:
        return header + dd + footer
    return ""
