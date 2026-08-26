"""searchhub.py — 应用侧统一搜索层 v2.

所有子模型基于同一份搜索材料作答(对比变量干净)。

策略(anysearch, 打满三成功力):
  1. batch_search 双通道并发: 通用查询 + finance 垂直查询(金融/时效类问题)
  2. extract 全文抓取: 挑最佳新闻结果抓正文(标题党多正文少,实数在正文里)
  3. ddgs 兜底(anysearch 未配 key 或失败时)

对外: search(question, cfg) -> str(拼接好的参考资料,空串=没搜到)
"""
import json
import logging
import re
from datetime import datetime, timezone, timedelta

import requests as _rq

logger = logging.getLogger("moa.searchhub")

_MCP_HEADERS = {"Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"}

# 金融/时效类问题走双通道
_FINANCE_PAT = re.compile(
    r"美股|股市|大盘|指数|道琼|纳斯达克|标普|A股|涨停|板块|股价|基金|汇率|"
    r"原油|黄金|期货|财报|市值|行情|收盘|开盘|IPO|加息|降息|美联储", re.IGNORECASE)
_TIME_PAT = re.compile(r"昨夜|昨晚|今天|今日|现在|最新|最近|刚刚|实时|当前|本周|本月")

_MCP_URL = "https://api.anysearch.com/mcp"


def _mcp_call(key: str, name: str, arguments: dict, timeout: int = 25) -> dict | None:
    """调 anysearch MCP 工具,返回 result dict 或 None。自动重试(SSL 偶发掐断)。"""
    last_err = None
    for attempt in range(3):
        try:
            r = _rq.post(_MCP_URL, json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                         "params": {"name": name, "arguments": arguments}},
                         headers={**_MCP_HEADERS, "Authorization": f"Bearer {key}"},
                         timeout=timeout)
            if r.status_code != 200:
                logger.warning("anysearch %s HTTP %s", name, r.status_code)
                return None
            data = r.json()
            if "error" in data:
                logger.warning("anysearch %s error: %s", name, data["error"])
                return None
            return data.get("result") or {}
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            import time as _t
            _t.sleep(1.2 * (attempt + 1))  # 退避 1.2s/2.4s
    logger.warning("anysearch %s failed after 3 tries: %s", name, last_err)
    return None


def _content_text(result: dict | None) -> str:
    """MCP result.content[0].text 提取。"""
    if not result:
        return ""
    for c in result.get("content") or []:
        if isinstance(c, dict) and c.get("text"):
            return c["text"]
    return ""


# ---------------------------------------------------------------- anysearch v2
def _parse_results_md(md: str) -> list[dict]:
    """从 anysearch markdown 结果里解析出 {title, url, snippet, date}。"""
    out = []
    for m in re.finditer(r"###\s*\d+\.\s*(.+?)\n-\s*\*\*URL\*\*:\s*(\S+)\n\s*(.*?)(?=\n###|\Z)",
                         md, re.DOTALL):
        title, url, rest = m.group(1).strip(), m.group(2).strip(), m.group(3)
        dm = re.search(r"date:\s*([A-Za-z]+\s*\d{1,2},\s*\d{4})", rest)
        out.append({"title": title, "url": url, "snippet": rest.strip()[:300],
                    "date": dm.group(1) if dm else ""})
    return out


def _score_item(it: dict, question: str = "") -> int:
    """给搜索结果打分: 详情页+日期新鲜+snippet 带实数者优先。"""
    score = 0
    t = it.get("title", "") + it.get("snippet", "")
    url = it["url"]
    if re.search(r"article/detail|/news/|/detail/|\.shtml|/\d{6,}/", url):
        score += 3
    if "detail" in url or "article" in url:
        score += 2
    # 导航/频道/走势图/工具页强降权(数据型查询的毒药)
    if re.search(r"america\.html|/channel|/quote|finance\.sina|google\.com/finance|"
                 r"investing\.com/(indices|markets)|academy|xueqiu\.com/\d+/\d|zhihu\.com/question|"
                 r"trend|走势|kjxq|/kjxx/|list\.html|results/ssq$|/ssq$|fanyi|translate", url):
        score -= 6
    # 数据查询型问题(开奖/比分/价格): snippet 里带"日期+数字组合"的重奖
    is_data_query = bool(re.search(r"开奖|号码|比分|价格|多少|几号|结果|中出", question))
    d = it.get("date", "")
    if d:
        try:
            dt = datetime.strptime(d.replace("  ", " "), "%b %d, %Y")
            age_days = (datetime.now() - dt).days
            if age_days <= 1:
                score += 6
            elif age_days <= 7:
                score += 2
            elif age_days > 60:
                score -= 8
        except ValueError:
            pass
    if re.search(r"\d{2}[,.]?\d{2,}\s*点|涨\d|跌\d|\+\d+\.\d+%|-\d+\.\d+%", t):
        score += 2
    # 开奖号码特征: 日期+连续独立数字(02 13 14 16 20 24 05 这种)
    if is_data_query and re.search(r"20\d{2}-\d{2}-\d{2}.*\d{2}\s+\d{2}\s+\d{2}", t):
        score += 8
    if is_data_query and re.search(r"\d{2}\s\d{2}\s\d{2}\s\d{2}\s\d{2}\s\d{2}", t):
        score += 6
    return score


def _pick_extract_url(items: list[dict], today: str, question: str = "") -> str:
    """挑最值得抓全文的 URL。"""
    best = max(items, key=lambda it: _score_item(it, question), default=None)
    url = best["url"] if best else ""
    return url if _score_item(best or {}, question) >= 2 else ""


def _search_anysearch_v2(question: str, key: str) -> str:
    """batch_search 双通道(通用+垂直) + extract 最佳全文。"""
    is_finance = bool(_FINANCE_PAT.search(question))
    # 数据/时效类查询: query 里带上日期约束(大幅提升命中最新一期)
    q_dated = question
    if re.search(r"最新|昨|开奖|结果|号码|比分|今天|今日|当前|现在", question):
        q_dated = f"{question} {datetime.now().strftime('%Y年%m月%d日')}"
    queries = [{"query": q_dated, "max_results": 8}]
    if is_finance:
        # 垂直通道: get_sub_domains 动态拿合法 sub_domain(文档硬性要求,不能猜)
        sub = _mcp_call(key, "get_sub_domains", {"domains": ["finance"]})
        sub_text = _content_text(sub) or ""
        # 新闻类问题用 finance.news, 行情价格类用 finance.quote
        prefer = "finance.quote" if re.search(_TIME_PAT, question) and re.search(
            r"价格|多少点|点位|报价|行情", question) else "finance.news"
        sub_domain = ""
        for cand in (prefer, "finance.news", "finance.quote"):
            if cand in sub_text:
                sub_domain = cand
                break
        if sub_domain:
            queries.append({"query": q_dated, "domain": "finance",
                            "sub_domain": sub_domain, "max_results": 6})

    res = _mcp_call(key, "batch_search", {"queries": queries}, timeout=30)
    md_parts = []
    items = []
    ct = _content_text(res)
    if ct:
        md_parts.append(ct)
        items = _parse_results_md(ct)
    else:
        # batch_search 不可用则退回单查
        single = _mcp_call(key, "search", {"query": question, "max_results": 8})
        st = _content_text(single)
        if st:
            md_parts.append(st)
            items = _parse_results_md(st)

    # extract 最佳结果全文(失败静默,多候选轮询)
    if items:
        # 按评分排序取前3个候选
        ranked = sorted(items, key=lambda it: _score_item(it, question), reverse=True)
        for cand in ranked[:3]:
            url = cand["url"]
            ext = _mcp_call(key, "extract", {"url": url}, timeout=20)
            et = _content_text(ext)
            # 失败特征: extract_failed / Unable to extract
            if et and "extract_failed" not in et and "Unable to extract" not in et:
                et = re.sub(r"\n{3,}", "\n\n", et).strip()
                if len(et) > 3200:
                    et = et[:3200] + "\n…(全文截断)"
                md_parts.append(f"\n## 最佳来源全文({url})\n{et}")
                break

    return "\n\n".join(p for p in md_parts if p)


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
    # 当前时间锚点(关键: LLM 不知道今天几号,会把旧闻当"昨夜")
    now = datetime.now(timezone(timedelta(hours=8)))
    tz_str = now.strftime("%Y-%m-%d %H:%M")
    weekday = "一二三四五六日"[now.weekday()]
    header = (f"[当前时间: {tz_str} (星期{weekday}, 北京时间)]\n"
              f"[联网搜索资料 · 与问题「{question}」相关 · "
              f"注意甄别结果时效,优先采信日期最接近当前时间的条目]\n")
    footer = ("\n[以上为搜索参考资料,回答时可作为依据,不确定的可声明。"
              "条目日期若早于今天,只作背景,不要当作\"昨夜/今天\"的行情]")

    key = cfg.get("anysearch_key") or ""
    if key:
        md = _search_anysearch_v2(question, key)
        if md:
            return header + md + footer
        logger.info("anysearch v2 未返回,降级 ddgs")

    dd = _search_ddgs(question)
    if dd:
        return header + dd + footer
    return ""
