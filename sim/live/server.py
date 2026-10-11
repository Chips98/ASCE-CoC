"""实时监控页：仿真进程内嵌的轻量 HTTP + SSE 服务。

为什么用 SSE 而不是 WebSocket
-----------------------------
数据方向是单向的（服务端 → 浏览器），SSE 用标准库即可实现，不引入
flask-socketio / eventlet 这类依赖，也不需要改变进程模型。断线时浏览器
的 `EventSource` 会自动带 `Last-Event-ID` 重连，我们据此补发增量。

为什么读数据库而不是进程内队列
------------------------------
仿真本身已经把"每轮每人看到了什么、想了什么、做了什么"完整落库了
（`agent_turn` / `agent_chain_step` / `agent_tool_call`）。实时页直接读同一份
数据，就不会出现"页面看到的"与"库里存的"不一致；而且页面在仿真结束后
依然可用，可以直接当复盘工具。WAL 模式让读写并发不互相阻塞。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from ..platform import db as dbm

logger = logging.getLogger("asce.live")

STATIC_DIR = Path(__file__).with_name("static")


# ==========================================================================
# 数据查询
# ==========================================================================


class LiveData:
    """只读数据访问。每次请求新开连接，避免多线程共享 sqlite 连接。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def _conn(self):
        return dbm.connect(self.db_path, read_only=True)

    # ---------------------------------------------------------------- 概览
    def overview(self) -> Dict[str, Any]:
        conn = self._conn()
        try:
            def one(sql: str, params=()) -> int:
                row = dbm.fetch_one(conn, sql, params)
                return int(row[0]) if row and row[0] is not None else 0

            meta = {
                r["key"]: r["value"]
                for r in dbm.fetch_all(conn, "SELECT key, value FROM sim_meta")
            }
            max_tick = one("SELECT COALESCE(MAX(tick),0) FROM sim_tick_stat")
            tstats = [dict(r) for r in dbm.fetch_all(
                conn, "SELECT * FROM sim_tick_stat ORDER BY tick ASC"
            )]
            return {
                "topic": meta.get("topic", ""),
                "background": meta.get("background", ""),
                "model": meta.get("model", ""),
                "chain_profile": meta.get("chain_profile", ""),
                "num_agents": int(meta.get("num_agents") or 0),
                "total_ticks": int(meta.get("num_timesteps") or 0),
                "pool_size": int(meta.get("pool_size") or 0),
                "seed_pool_size": int(meta.get("seed_pool_size") or 0),
                "current_tick": max_tick,
                "totals": {
                    "users": one("SELECT COUNT(*) FROM user WHERE is_control=0"),
                    "posts": one("SELECT COUNT(*) FROM post WHERE step_number>0"),
                    "all_posts": one("SELECT COUNT(*) FROM post"),
                    "comments": one("SELECT COUNT(*) FROM comment"),
                    "likes": one("SELECT COUNT(*) FROM post_like"),
                    "follows": one("SELECT COUNT(*) FROM follow"),
                    "turns": one("SELECT COUNT(*) FROM agent_turn"),
                    "tool_calls": one("SELECT COUNT(*) FROM agent_tool_call"),
                    "tokens": one("SELECT COALESCE(SUM(prompt_tokens+completion_tokens),0) FROM agent_turn"),
                },
                "ticks": tstats,
            }
        finally:
            conn.close()

    # ---------------------------------------------------------------- 活动流
    def activities(self, tick: Optional[int], limit: int = 60) -> List[Dict[str, Any]]:
        conn = self._conn()
        try:
            params: Tuple[Any, ...] = ()
            where = "WHERE t.is_active=1"
            if tick is not None:
                where += " AND t.tick=?"
                params = (int(tick),)
            rows = dbm.fetch_all(
                conn,
                f"""SELECT t.agent_id, t.tick, t.status, t.primary_action, t.primary_target_id,
                           t.reason, t.thought_text, t.stage1_text, t.stage2_text,
                           t.chain_steps_json, t.actions_json, t.feed_post_ids,
                           t.stage1_ms, t.stage2_ms, t.prompt_tokens, t.completion_tokens,
                           u.name AS agent_name, u.persona AS agent_persona
                    FROM agent_turn t
                    LEFT JOIN user u ON u.user_id = t.agent_id
                    {where}
                    ORDER BY t.tick DESC, t.agent_id ASC
                    LIMIT ?""",
                params + (int(limit),),
            )
            out: List[Dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                d["feed_post_ids"] = _loads(d.get("feed_post_ids"), [])
                d["actions"] = _loads(d.get("actions_json"), [])
                d["chain"] = _parse_chain_brief(d.get("chain_steps_json"))
                d.pop("actions_json", None)
                d.pop("chain_steps_json", None)
                out.append(d)
            return out
        finally:
            conn.close()

    # ---------------------------------------------------------------- 认知链健康度
    def chain_health(self, tick: Optional[int]) -> List[Dict[str, Any]]:
        """各节点的写出率。缺节点会立刻暴露"模型没按契约输出"。"""
        conn = self._conn()
        try:
            if tick is None:
                row = dbm.fetch_one(conn, "SELECT COALESCE(MAX(tick),0) FROM sim_tick_stat")
                tick = int(row[0]) if row else 0
            rows = dbm.fetch_all(
                conn,
                """SELECT dim_label, node_label, node_key,
                          SUM(present) AS present_n,
                          SUM(CASE WHEN present=1 AND COALESCE(inherited,0)=0 THEN 1 ELSE 0 END) AS explicit_n,
                          COUNT(*) AS total_n
                   FROM agent_chain_step WHERE tick=?
                   GROUP BY dim_label, node_label, node_key
                   ORDER BY MIN(stage), MIN(step_order)""",
                (int(tick),),
            )
            out = []
            for r in rows:
                total = max(int(r["total_n"] or 1), 1)
                pres = int(r["present_n"] or 0)
                exp = int(r["explicit_n"] or 0)
                out.append({
                    "dim": r["dim_label"],
                    "node": r["node_label"],
                    "key": r["node_key"],
                    "present": pres,
                    # explicit = 模型把节点名写出来的比例；present 含维度级兜底
                    "explicit": exp,
                    "total": total,
                    "rate": round(pres / total, 3),
                    "explicit_rate": round(exp / total, 3),
                })
            return out
        finally:
            conn.close()

    # ---------------------------------------------------------------- 工具分布
    def tool_stats(self, tick: Optional[int]) -> List[Dict[str, Any]]:
        conn = self._conn()
        try:
            if tick is None:
                rows = dbm.fetch_all(
                    conn,
                    """SELECT tool_name, tool_cn, COUNT(*) AS n, SUM(success) AS ok_n
                       FROM agent_tool_call GROUP BY tool_name, tool_cn ORDER BY n DESC""",
                )
            else:
                rows = dbm.fetch_all(
                    conn,
                    """SELECT tool_name, tool_cn, COUNT(*) AS n, SUM(success) AS ok_n
                       FROM agent_tool_call WHERE tick=?
                       GROUP BY tool_name, tool_cn ORDER BY n DESC""",
                    (int(tick),),
                )
            return [
                {"name": r["tool_name"], "cn": r["tool_cn"], "n": int(r["n"]),
                 "ok": int(r["ok_n"] or 0)} for r in rows
            ]
        finally:
            conn.close()

    # ---------------------------------------------------------------- 单条完整思考
    def turn_detail(self, agent_id: int, tick: int) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        try:
            r = dbm.fetch_one(
                conn,
                """SELECT t.*, u.name AS agent_name, u.persona AS agent_persona, u.bio AS agent_bio
                   FROM agent_turn t LEFT JOIN user u ON u.user_id=t.agent_id
                   WHERE t.agent_id=? AND t.tick=?""",
                (int(agent_id), int(tick)),
            )
            if r is None:
                return None
            d = dict(r)
            d["chain"] = _parse_chain_full(d.get("chain_steps_json"))
            d["actions"] = _loads(d.get("actions_json"), [])
            d["feed_post_ids"] = _loads(d.get("feed_post_ids"), [])
            d["feed_posts"] = self._feed_posts(conn, d["feed_post_ids"])
            return d
        finally:
            conn.close()

    @staticmethod
    def _feed_posts(conn, ids: List[int]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = dbm.fetch_all(
            conn,
            f"""SELECT p.post_id, p.user_id, p.content, p.num_likes, p.num_dislikes,
                       u.name AS author_name
                FROM post p LEFT JOIN user u ON u.user_id=p.user_id
                WHERE p.post_id IN ({marks})""",
            tuple(ids),
        )
        return [dict(r) for r in rows]


def _loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _parse_chain_brief(raw: Any) -> List[Dict[str, Any]]:
    """把 chain_steps_json 压成前端好用的扁平列表。"""
    obj = _loads(raw, {}) or {}
    out: List[Dict[str, Any]] = []
    for stage_key, stage_no in (("stage1", 1), ("stage2", 2)):
        st = obj.get(stage_key) or {}
        for s in st.get("steps") or []:
            out.append({
                "stage": stage_no,
                "dim": s.get("dim_label"),
                "node": s.get("label"),
                "key": s.get("key"),
                "text": s.get("text") or "",
                "present": bool(s.get("present")),
                # inherited=True 表示内容来自「维度段兜底」（模型没标节点名），
                # 前端与打印页都要把这个区别显示出来，否则节点写出率会被高估。
                "inherited": bool(s.get("inherited")),
            })
    return out


def _parse_chain_full(raw: Any) -> Dict[str, Any]:
    obj = _loads(raw, {}) or {}
    return {
        "raw": obj,
        "steps": _parse_chain_brief(raw),
        "missing": (obj.get("stage1") or {}).get("missing", [])
        + (obj.get("stage2") or {}).get("missing", []),
    }


# ==========================================================================
# PDF 导出用的打印页
# ==========================================================================
# 用「服务端渲染一张打印友好的页面 + 浏览器打印」实现 PDF 导出，而不是引入
# PDF 生成库。原因：正文是中文，Python 端生成 PDF 需要内嵌 CJK 字体（体积与
# 授权都是负担），而浏览器自带完整的中文字体与分页控制，输出质量更好，
# 并且零新增依赖、离线可用。


#: 工具名 -> 中文名。打印页要把动作名显示成中文（前端从接口拿到的已经是中文，
#: 但 agent_turn.primary_action 存的是原始工具名）。
_ACTION_CN = {
    "like_post": "点赞帖子", "dislike_post": "点踩帖子", "create_comment": "发表评论",
    "reply_to_comment": "回复评论", "share_comment": "转发评论",
    "like_comment": "点赞评论", "dislike_comment": "点踩评论",
    "create_post": "发布新帖", "follow": "关注用户", "unfollow": "取消关注",
    "do_nothing": "不做动作", "refresh": "刷新信息流", "trend": "查看热门",
    "search_posts": "搜索帖子", "search_user": "搜索用户",
    "get_trending_comments": "查看热门评论", "get_post_detail": "查看帖子详情",
    "get_user_profile": "查看用户资料",
}


def _action_cn(name: Any) -> str:
    if not name:
        return "无"
    return _ACTION_CN.get(str(name), str(name))


def _esc(value: Any) -> str:
    """HTML 转义。模型输出里可能有 < > & 等字符。"""
    return (
        str(value if value is not None else "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _dimension_order() -> List[str]:
    try:
        from sim.chain.spec import DIMENSIONS
        return [d.label for d in DIMENSIONS]
    except Exception:
        return ["情绪", "情感", "思维", "立场", "观点", "意图", "一致性"]


def _dimension_theory() -> Dict[str, str]:
    try:
        from sim.chain.spec import DIMENSIONS
        return {d.label: d.theory for d in DIMENSIONS}
    except Exception:
        return {}


PRINT_CSS = """
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  background: #fff; color: #111827;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Hiragino Sans GB", "Microsoft YaHei", serif;
  font-size: 10.5pt; line-height: 1.7;
}
.wrap { max-width: 190mm; margin: 0 auto; padding: 12mm 8mm; }

.toolbar {
  position: sticky; top: 0; z-index: 5;
  display: flex; justify-content: flex-end; gap: 8px;
  padding: 8px 0 14px; margin-bottom: 4px;
  background: #fff; border-bottom: 1px solid #e5e7eb;
}
.toolbar button {
  font: inherit; font-size: 9.5pt;
  padding: 5px 14px; border-radius: 6px; cursor: pointer;
  border: 1px solid #d1d5db; background: #fff; color: #374151;
}
.toolbar button.primary { background: #1d4ed8; border-color: #1d4ed8; color: #fff; font-weight: 600; }

h1 { margin: 0 0 2px; font-size: 17pt; letter-spacing: -.2px; }
.subtitle { margin: 0 0 14px; font-size: 9.5pt; color: #6b7280; }

table.meta { width: 100%; border-collapse: collapse; margin-bottom: 18px; font-size: 9.5pt; }
table.meta td { padding: 4px 8px; border: 1px solid #e5e7eb; }
table.meta td.k { background: #f9fafb; color: #6b7280; width: 64px; white-space: nowrap; }
table.meta td.v { font-weight: 600; }

h2 {
  font-size: 12pt; margin: 22px 0 9px; padding-bottom: 5px;
  border-bottom: 1.5px solid #111827;
  page-break-after: avoid;
}
h2 .n { font-size: 9pt; color: #6b7280; font-weight: 400; margin-left: 6px; }

.post {
  border: 1px solid #e5e7eb; border-radius: 5px;
  padding: 8px 11px; margin-bottom: 7px;
  page-break-inside: avoid;
}
.post-head { font-size: 8.5pt; color: #6b7280; margin-bottom: 3px; }
.post-head b { color: #1d4ed8; }
.post-body { white-space: pre-wrap; word-break: break-word; }

.dim { margin-bottom: 15px; page-break-inside: avoid; }
.dim-head {
  display: flex; align-items: baseline; gap: 8px;
  font-size: 10.5pt; font-weight: 700;
  padding-bottom: 4px; margin-bottom: 7px;
  border-bottom: 1px solid #e5e7eb;
}
.dim-head .theory { font-weight: 400; font-size: 8.5pt; color: #9ca3af; margin-left: auto; }

.node { margin-bottom: 9px; page-break-inside: avoid; }
.node-line { font-size: 9pt; color: #6b7280; margin-bottom: 2px; }
.node-key {
  display: inline-block; font-weight: 700; color: #374151;
  border: 1px solid #e5e7eb; background: #f9fafb;
  border-radius: 3px; padding: 0 4px; margin-right: 6px;
}
.node-label { font-weight: 600; color: #374151; }
.node-text { white-space: pre-wrap; word-break: break-word; }
.node.missing .node-text { color: #b91c1c; font-style: italic; }
.tag-inherited {
  font-size: 7.5pt; color: #b45309; border: 1px solid #fcd9a8;
  background: #fef6ec; border-radius: 3px; padding: 0 3px; margin-left: 5px;
}

.act {
  border: 1px solid #e5e7eb; border-radius: 5px;
  padding: 8px 11px; margin-bottom: 7px;
  page-break-inside: avoid;
}
.act.ok  { background: #f4fbf7; }
.act.err { background: #fdf5f4; }
.act-head { font-weight: 700; font-size: 10pt; }
.act-order { color: #6b7280; font-weight: 400; margin-right: 6px; }
.act-args { font-size: 9pt; color: #4b5563; margin-top: 4px; word-break: break-word; }
.act-args code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 9pt; background: #f3f4f6; border-radius: 3px; padding: 1px 4px;
}
.act-err { font-size: 9pt; color: #b91c1c; margin-top: 4px; }

.raw {
  border: 1px solid #e5e7eb; border-radius: 5px;
  padding: 10px 12px; margin-bottom: 12px;
  white-space: pre-wrap; word-break: break-word;
  background: #fcfcfd;
}
.raw h3 { margin: 0 0 6px; font-size: 10pt; }
.raw.avoid { page-break-inside: avoid; }

.foot {
  margin-top: 22px; padding-top: 10px;
  border-top: 1px solid #e5e7eb;
  font-size: 8.5pt; color: #9ca3af;
}

@page { size: A4; margin: 14mm 12mm; }
@media print {
  .toolbar { display: none !important; }
  .wrap { max-width: none; padding: 0; }
  body { font-size: 10pt; }
  h2 { break-after: avoid; }
  .dim, .node, .act, .post { break-inside: avoid; }
}
"""


def render_print_page(detail: Dict[str, Any], meta: Dict[str, Any]) -> str:
    """把一条 turn 记录渲染成打印友好的页面（用于「导出 PDF」）。"""
    steps = (detail.get("chain") or {}).get("steps") or []
    missing = [s for s in steps if not s.get("present")]
    theory = _dimension_theory()

    # 按维度分组并保持固定顺序
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in steps:
        groups.setdefault(s.get("dim") or "未分组", []).append(s)
    ordered = [d for d in _dimension_order() if d in groups]
    ordered += [d for d in groups if d not in ordered]

    topic = meta.get("topic") or ""
    topic_cn = "中美关税战" if topic == "cut" else topic
    agent_name = detail.get("agent_name") or ("用户" + str(detail.get("agent_id")))
    tick = detail.get("tick")
    total_ticks = meta.get("total_ticks") or ""
    status = detail.get("status") or ""
    status_cn = {"ok": "正常", "failed": "失败", "partial": "部分完成",
                 "fallback": "兜底", "skipped": "未激活"}.get(status, status)

    parts: List[str] = []
    A = parts.append

    A("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
    A(f"<title>认知链记录 · 用户{_esc(detail.get('agent_id'))} · 第{_esc(tick)}轮</title>")
    A(f"<style>{PRINT_CSS}</style></head><body><div class='wrap'>")

    A("<div class='toolbar'>")
    A("<button class='primary' onclick='window.print()'>打印 / 保存为 PDF</button>")
    A("<button onclick='window.close()'>关闭</button>")
    A("</div>")

    A("<h1>认知链推理记录</h1>")
    A(f"<p class='subtitle'>{_esc(topic_cn)} · "
      f"{_esc(meta.get('chain_profile') or '')} 档认知链 · "
      f"{_esc(meta.get('model') or '')}</p>")

    A("<table class='meta'>")
    A(f"<tr><td class='k'>智能体</td><td class='v'>{_esc(agent_name)}</td>"
      f"<td class='k'>编号</td><td class='v'>{_esc(detail.get('agent_id'))}</td>"
      f"<td class='k'>轮次</td><td class='v'>第 {_esc(tick)} / {_esc(total_ticks)} 轮</td></tr>")
    A(f"<tr><td class='k'>状态</td><td class='v'>{_esc(status_cn)}</td>"
      f"<td class='k'>主导动作</td><td class='v'>{_esc(_action_cn(detail.get('primary_action')))}"
      + (f" #{_esc(detail.get('primary_target_id'))}" if detail.get("primary_target_id") else "")
      + "</td>"
      f"<td class='k'>节点完整度</td><td class='v'>{len(steps) - len(missing)} / {len(steps)}</td></tr>")
    A(f"<tr><td class='k'>思考耗时</td><td class='v'>"
      f"{((detail.get('stage1_ms') or 0) + (detail.get('stage2_ms') or 0)) / 1000:.1f} 秒</td>"
      f"<td class='k'>token</td><td class='v'>"
      f"{_esc((detail.get('prompt_tokens') or 0) + (detail.get('completion_tokens') or 0))}</td>"
      f"<td class='k'>记录时间</td><td class='v'>{_esc(detail.get('created_at') or '')}</td></tr>")
    A("</table>")

    # ① 看到的内容
    posts = detail.get("feed_posts") or []
    A(f"<h2>① 他看到了什么<span class='n'>{len(posts)} 条帖子</span></h2>")
    if posts:
        for p in posts:
            A("<div class='post'>")
            A(f"<div class='post-head'><b>帖子 {_esc(p.get('post_id'))}</b> · "
              f"{_esc(p.get('author_name') or ('用户' + str(p.get('user_id'))))} · "
              f"赞 {_esc(p.get('num_likes'))} · 踩 {_esc(p.get('num_dislikes'))}</div>")
            A(f"<div class='post-body'>{_esc(p.get('content') or '')}</div>")
            A("</div>")
    else:
        A("<p class='node-text'>本轮没有刷到内容。</p>")

    # ② 认知链
    A(f"<h2>② 认知链<span class='n'>{len(steps)} 个节点"
      + (f" · 缺 {len(missing)}" if missing else "") + "</span></h2>")
    for dim in ordered:
        A("<div class='dim'>")
        A(f"<div class='dim-head'><span>{_esc(dim)}</span>"
          f"<span class='theory'>{_esc(theory.get(dim, ''))}</span></div>")
        for s in groups[dim]:
            cls = "node" + ("" if s.get("present") else " missing")
            A(f"<div class='{cls}'>")
            line = (f"<div class='node-line'><span class='node-key'>{_esc(s.get('key'))}</span>"
                    f"<span class='node-label'>{_esc(s.get('node'))}</span>")
            if s.get("present") and s.get("inherited"):
                line += "<span class='tag-inherited'>来自维度段</span>"
            line += "</div>"
            A(line)
            txt = s.get("text") if s.get("present") else "（模型未写出该节点）"
            A(f"<div class='node-text'>{_esc(txt)}</div>")
            A("</div>")
        A("</div>")

    # ③ 实际动作
    actions = detail.get("actions") or []
    A(f"<h2>③ 实际动作<span class='n'>{len(actions)} 次工具调用</span></h2>")
    if actions:
        for a in actions:
            ok = bool(a.get("success"))
            A(f"<div class='act {'ok' if ok else 'err'}'>")
            A(f"<div class='act-head'><span class='act-order'>#{_esc(a.get('order'))}</span>"
              f"{_esc(a.get('tool_cn') or a.get('tool'))}"
              + ("" if ok else " · 未执行") + "</div>")
            A("<div class='act-args'>参数：<code>"
              f"{_esc(json.dumps(a.get('args') or {}, ensure_ascii=False))}</code></div>")
            if not ok and a.get("error"):
                A(f"<div class='act-err'>拒绝原因：{_esc(a.get('error'))}</div>")
            A("</div>")
    else:
        A("<p class='node-text'>本轮没有调用任何工具。</p>")

    # ④ 两阶段原文
    A("<h2>④ 两阶段原文</h2>")
    A("<div class='raw avoid'><h3>第一阶段 · 感受与自我（情绪 / 情感 / 思维 / 立场）</h3>"
      f"{_esc(detail.get('stage1_text') or '（无输出）')}</div>")
    A("<div class='raw'><h3>第二阶段 · 判断与行动（观点 / 意图 / 一致性）</h3>"
      f"{_esc(detail.get('stage2_text') or '（无输出）')}</div>")
    if detail.get("error_text"):
        A("<div class='raw'><h3>执行异常</h3>"
          f"{_esc(detail.get('error_text'))}</div>")

    A("<div class='foot'>由 ASCE-CoC 仿真监控工作台导出 · "
      "本页内容全部来自仿真数据库记录，未经任何二次改写。</div>")
    A("</div>")

    # 打开即弹出打印对话框，用户选择「另存为 PDF」即可
    A("<script>window.addEventListener('load',function(){"
      "setTimeout(function(){try{window.print()}catch(e){}},350)});</script>")
    A("</body></html>")
    return "".join(parts)


# ==========================================================================
# HTTP 处理
# ==========================================================================


class _Handler(BaseHTTPRequestHandler):
    data: LiveData = None  # type: ignore[assignment]
    server_version = "ASCE-CoC-Live/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # 静默 access log
        return

    # ------------------------------------------------------------------ 工具
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _file(self, name: str, ctype: str) -> None:
        path = STATIC_DIR / name
        if not path.exists():
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        self._send(200, path.read_bytes(), ctype)

    # ------------------------------------------------------------------ 路由
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        def qint(key: str) -> Optional[int]:
            v = qs.get(key)
            if not v or v[0] in ("", "null", "undefined"):
                return None
            try:
                return int(v[0])
            except ValueError:
                return None

        try:
            if path in ("/", "/index.html"):
                self._file("index.html", "text/html; charset=utf-8")
            elif path == "/static/app.js":
                self._file("app.js", "application/javascript; charset=utf-8")
            elif path == "/static/app.css":
                self._file("app.css", "text/css; charset=utf-8")
            elif path == "/api/overview":
                self._json(self.data.overview())
            elif path == "/api/activities":
                self._json(self.data.activities(qint("tick"), qint("limit") or 60))
            elif path == "/api/chain-health":
                self._json(self.data.chain_health(qint("tick")))
            elif path == "/api/tool-stats":
                self._json(self.data.tool_stats(qint("tick")))
            elif path == "/api/turn":
                aid, tk = qint("agent_id"), qint("tick")
                if aid is None or tk is None:
                    self._json({"error": "需要 agent_id 与 tick"}, 400)
                else:
                    detail = self.data.turn_detail(aid, tk)
                    self._json(detail or {"error": "未找到该轮记录"}, 200 if detail else 404)
            elif path == "/print/turn":
                aid, tk = qint("agent_id"), qint("tick")
                if aid is None or tk is None:
                    self._send(400, "需要 agent_id 与 tick".encode("utf-8"),
                               "text/plain; charset=utf-8")
                else:
                    detail = self.data.turn_detail(aid, tk)
                    if detail is None:
                        self._send(404, f"未找到 用户{aid} 第{tk}轮 的记录".encode("utf-8"),
                                   "text/plain; charset=utf-8")
                    else:
                        meta = self.data.overview()
                        html = render_print_page(detail, meta)
                        self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/stream":
                self._stream(qint("from"))
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # 页面永不因单次查询失败而整体挂掉
            logger.exception("live 请求失败 %s", path)
            try:
                self._json({"error": str(exc)}, 500)
            except Exception:
                pass

    # ------------------------------------------------------------------ SSE
    def _stream(self, from_tick: Optional[int]) -> None:
        """推送 `snapshot` / `tick` / `activity` / `chain` 四类事件。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        last_tick = int(from_tick or 0)
        last_turn_key = ""
        sent_ids: set = set()
        first = True

        while True:
            try:
                ov = self.data.overview()
            except Exception:
                time.sleep(1.0)
                continue

            cur = int(ov.get("current_tick") or 0)

            if first:
                # 首帧：全量快照 + 已有活动流，让页面一打开就有内容。
                # 顺序很重要：先推**当前轮**的活动（前端默认"只看当前轮"，
                # 若先推历史轮次会被前端过滤掉，页面看起来是空的），
                # 再补最近的活动作为上下文。
                self._emit("snapshot", ov)
                cur_list = self.data.activities(cur, 80) if cur else []
                for a in reversed(cur_list):
                    key = f"{a['agent_id']}:{a['tick']}"
                    sent_ids.add(key)
                    self._emit("activity", a)
                for a in reversed(self.data.activities(None, 40)):
                    key = f"{a['agent_id']}:{a['tick']}"
                    if key in sent_ids:
                        continue
                    sent_ids.add(key)
                    self._emit("activity", a)
                self._emit("chain", {"tick": cur, "nodes": self.data.chain_health(None)})
                self._emit("tools", {"tick": cur, "items": self.data.tool_stats(None)})
                first = False
                last_tick = cur
            elif cur != last_tick:
                self._emit("snapshot", ov)
                for a in reversed(self.data.activities(cur, 80)):
                    key = f"{a['agent_id']}:{a['tick']}"
                    if key not in sent_ids:
                        sent_ids.add(key)
                        self._emit("activity", a)
                self._emit("chain", {"tick": cur, "nodes": self.data.chain_health(cur)})
                self._emit("tools", {"tick": cur, "items": self.data.tool_stats(cur)})
                last_tick = cur
            else:
                # 同一 tick 内也持续推新完成的 turn
                for a in reversed(self.data.activities(cur, 80)):
                    key = f"{a['agent_id']}:{a['tick']}"
                    if key not in sent_ids:
                        sent_ids.add(key)
                        self._emit("activity", a)

            self._emit("ping", {"t": time.time()})
            time.sleep(0.8)

    def _emit(self, event: str, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str)
        self.wfile.write(f"event: {event}\ndata: {body}\n\n".encode("utf-8"))
        self.wfile.flush()


class LiveServer:
    """独立进程里跑的实时监控页。

    为什么用子进程而不是线程：仿真进程的主循环长期被 await 占着，而且仿真
    结束（或崩溃）后用户仍需要能打开页面复盘。独立进程同时解决这两点，也避免
    了"以脚本方式运行时相对导入越界"的问题。
    """

    def __init__(self, host: str, port: int, db_path: str) -> None:
        self.host = host
        self.port = int(port)
        self.db_path = db_path
        self.proc = None
        self.log_path = ""
        self.url = f"http://{host}:{self.port}/"

    def start(self) -> None:
        import subprocess

        # server.py 位于 <项目根>/sim/live/server.py，所以项目根是 parents[2]。
        # 用 parent.parent.parent 会指到项目根之上，导致子进程里
        # `-m sim.live.server` 找不到包（实测报
        # "attempted relative import beyond top-level package"）。
        root = Path(__file__).resolve().parents[2]
        # 日志文件名带上端口，避免同一库多次运行时读到上一次的失败日志
        # （之前正是因为复用了固定文件名，把旧的报错当成了新进程的报错，
        #   浪费了一轮排查）。
        self.log_path = str(Path(self.db_path).with_suffix(f".live{self.port}.log"))
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        # 先把日志文件截断，确保读到的任何内容都属于本次启动
        log = open(self.log_path, "w", encoding="utf-8")
        log.write(f"[launcher] root={root} port={self.port}\n")
        log.flush()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "sim.live.server",
             "--db", self.db_path, "--host", self.host, "--port", str(self.port)],
            cwd=str(root),
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        )
        # 等端口就绪，最多 8 秒
        import socket

        for _ in range(32):
            time.sleep(0.25)
            if self.proc.poll() is not None:
                tail = ""
                try:
                    tail = Path(self.log_path).read_text(encoding="utf-8")[-500:]
                except OSError:
                    pass
                raise RuntimeError(f"实时页进程立即退出（code={self.proc.returncode}）：{tail}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                if s.connect_ex((self.host, self.port)) == 0:
                    logger.info("实时监控页已启动：%s", self.url)
                    return
        raise RuntimeError(f"实时页端口 {self.port} 在 8 秒内未就绪，日志：{self.log_path}")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


def start_live_server(db_path: str, live_cfg: Dict[str, Any], _conn=None) -> LiveServer:
    """供仿真主流程调用。`_conn` 仅为显式声明"读写同一个库"这一事实。"""
    host = str(live_cfg.get("host") or "127.0.0.1")
    port = int(live_cfg.get("port") or 8801)
    srv = LiveServer(host, port, db_path)
    srv.start()
    return srv


def main() -> None:
    """独立启动实时页（用于查看已跑完的库）。

    用法：python -m sim.live.server --db runs/xxx.db --port 8801
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8801)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    data = LiveData(os.path.abspath(args.db))
    handler = type("BoundHandler", (_Handler,), {"data": data})
    httpd = ThreadingHTTPServer((args.host, int(args.port)), handler)
    httpd.daemon_threads = True
    print(f"ASCE-CoC 实时监控页：http://{args.host}:{args.port}/  库={args.db}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
