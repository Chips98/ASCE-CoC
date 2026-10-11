"""认知链落库。

表结构分工
----------
* `agent_turn`：每轮每 agent 一行的宽记录。回答"这一轮你看到了什么、想了什么、
  做了什么、为什么"。把三类信息放在同一行是有意的：若拆进不同的表、只用
  `(agent_id, tick)` 隐式关联，分析时要写多路 LEFT JOIN，而且无法保证各表
  记录的是同一轮同一状态。这里合并成一张宽表 + 两张明细表。
* `agent_chain_step`：认知链的节点级明细（长表）。做分布统计、机理分析、
  "某个节点是否总是被跳过"这类检查时用它，比在 JSON 里挖方便得多。
* `agent_tool_call`：工具调用明细。一个 turn 可以有多次调用，包括被拒绝的，
  这是分析"模型为什么没做成事"的关键证据。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..platform import db as dbm

logger = logging.getLogger("asce.store")


class TurnStore:
    """认知链与工具调用的持久化。"""

    def __init__(self, conn) -> None:
        self.conn = conn

    # ==================================================================
    # 写
    # ==================================================================
    def save_turn(
        self,
        agent_id: int,
        tick: int,
        is_active: bool,
        status: str,
        error: str,
        feed_post_ids: Sequence[int],
        trending_ids: Sequence[int],
        interaction_ids: Sequence[int],
        thought_text: str,
        stage1_text: str,
        stage2_text: str,
        chain_steps: Dict[str, Any],
        chains: Sequence[Tuple[int, Any]],
        tool_calls: Sequence[Any],
        primary_action: str,
        primary_target: Optional[int],
        reason: str,
        profile: str,
        stage1_ms: int,
        stage2_ms: int,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> None:
        dbm.execute(
            self.conn,
            """INSERT INTO agent_turn(
                   agent_id,tick,is_active,status,error_text,
                   feed_post_ids,feed_trending_ids,interaction_ids,
                   thought_text,stage1_text,stage2_text,chain_steps_json,
                   actions_json,primary_action,primary_target_id,reason,
                   profile,stage1_ms,stage2_ms,prompt_tokens,completion_tokens,model)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(agent_id,tick) DO UPDATE SET
                   is_active=excluded.is_active, status=excluded.status,
                   error_text=excluded.error_text,
                   feed_post_ids=excluded.feed_post_ids,
                   feed_trending_ids=excluded.feed_trending_ids,
                   interaction_ids=excluded.interaction_ids,
                   thought_text=excluded.thought_text,
                   stage1_text=excluded.stage1_text,
                   stage2_text=excluded.stage2_text,
                   chain_steps_json=excluded.chain_steps_json,
                   actions_json=excluded.actions_json,
                   primary_action=excluded.primary_action,
                   primary_target_id=excluded.primary_target_id,
                   reason=excluded.reason, profile=excluded.profile,
                   stage1_ms=excluded.stage1_ms, stage2_ms=excluded.stage2_ms,
                   prompt_tokens=excluded.prompt_tokens,
                   completion_tokens=excluded.completion_tokens,
                   model=excluded.model""",
            (
                int(agent_id), int(tick), 1 if is_active else 0, status, error or None,
                _dumps(list(feed_post_ids)), _dumps(list(trending_ids)), _dumps(list(interaction_ids)),
                thought_text or None, stage1_text or None, stage2_text or None,
                _dumps(chain_steps),
                _dumps([_call_to_json(c) for c in tool_calls]),
                primary_action or None, primary_target, reason or None,
                profile, int(stage1_ms), int(stage2_ms),
                int(prompt_tokens), int(completion_tokens), model or None,
            ),
        )

        # 节点明细
        rows: List[Sequence[Any]] = []
        for stage, chain in chains:
            if chain is None:
                continue
            for order, step in enumerate(chain.steps):
                rows.append((
                    int(agent_id), int(tick), int(stage), step.dim, step.dim_label,
                    step.key, step.label, order, step.text or "",
                    1 if step.present else 0, 1 if step.inherited else 0,
                ))
        if rows:
            dbm.executemany(
                self.conn,
                """INSERT OR REPLACE INTO agent_chain_step(
                       agent_id,tick,stage,dim,dim_label,node_key,node_label,
                       step_order,text,present,inherited)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

        # 工具调用明细
        calls: List[Sequence[Any]] = []
        for i, c in enumerate(tool_calls, 1):
            row = c.to_row() if hasattr(c, "to_row") else dict(c)
            calls.append((
                int(agent_id), int(tick), int(row.get("call_order") or i),
                row.get("tool_name") or "", row.get("tool_cn") or "",
                _dumps(row.get("arguments_json")), _dumps(row.get("result_json")),
                int(row.get("success") or 0), row.get("error_text"),
            ))
        if calls:
            dbm.executemany(
                self.conn,
                """INSERT OR REPLACE INTO agent_tool_call(
                       agent_id,tick,call_order,tool_name,tool_cn,
                       arguments_json,result_json,success,error_text)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                calls,
            )

    def save_interactions(self, agent_id: int, tick: int, items: Sequence[Dict[str, Any]]) -> None:
        rows: List[Sequence[Any]] = []
        for i, it in enumerate(items, 1):
            rows.append((
                int(agent_id), int(tick), int(it.get("interaction_id") or i),
                it.get("from_id"), it.get("from_name") or it.get("from_persona"),
                it.get("post_id"), it.get("content"),
            ))
        if rows:
            dbm.executemany(
                self.conn,
                """INSERT OR REPLACE INTO agent_interaction(
                       agent_id,tick,interaction_id,from_id,from_name,post_id,content)
                   VALUES(?,?,?,?,?,?,?)""",
                rows,
            )

    def save_tick_stat(self, tick: int, stat: Dict[str, Any]) -> None:
        dbm.execute(
            self.conn,
            """INSERT INTO sim_tick_stat(
                   tick,active_count,post_count,comment_count,like_count,follow_count,
                   tool_call_count,chain_missing_count,stage1_ms_total,stage2_ms_total,
                   prompt_tokens,completion_tokens,elapsed_sec)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(tick) DO UPDATE SET
                   active_count=excluded.active_count, post_count=excluded.post_count,
                   comment_count=excluded.comment_count, like_count=excluded.like_count,
                   follow_count=excluded.follow_count,
                   tool_call_count=excluded.tool_call_count,
                   chain_missing_count=excluded.chain_missing_count,
                   stage1_ms_total=excluded.stage1_ms_total,
                   stage2_ms_total=excluded.stage2_ms_total,
                   prompt_tokens=excluded.prompt_tokens,
                   completion_tokens=excluded.completion_tokens,
                   elapsed_sec=excluded.elapsed_sec""",
            (
                int(tick), int(stat.get("active_count") or 0),
                int(stat.get("post_count") or 0), int(stat.get("comment_count") or 0),
                int(stat.get("like_count") or 0), int(stat.get("follow_count") or 0),
                int(stat.get("tool_call_count") or 0),
                int(stat.get("chain_missing_count") or 0),
                int(stat.get("stage1_ms_total") or 0),
                int(stat.get("stage2_ms_total") or 0),
                int(stat.get("prompt_tokens") or 0),
                int(stat.get("completion_tokens") or 0),
                float(stat.get("elapsed_sec") or 0.0),
            ),
        )

    # ==================================================================
    # 读（供 agent 记忆与实时页）
    # ==================================================================
    def recent_actions(self, agent_id: int, before_tick: int, rounds: int = 3) -> List[Dict[str, Any]]:
        """取最近 `rounds` 轮自己的动作与理由。"""
        rows = dbm.fetch_all(
            self.conn,
            """SELECT tick, is_active, reason, actions_json
               FROM agent_turn
               WHERE agent_id=? AND tick < ? AND tick > ?
               ORDER BY tick ASC""",
            (int(agent_id), int(before_tick), max(0, int(before_tick) - int(rounds))),
        )
        out: List[Dict[str, Any]] = []
        for r in rows:
            names: List[str] = []
            try:
                for c in json.loads(r["actions_json"] or "[]"):
                    if c.get("success"):
                        names.append(c.get("tool_cn") or c.get("tool_name") or "")
            except (TypeError, ValueError):
                pass
            out.append({
                "tick": int(r["tick"]),
                "is_active": bool(r["is_active"]),
                "reason": r["reason"] or "",
                "action_names": [n for n in names if n],
            })
        return out

    def platform_brief(self) -> Dict[str, Any]:
        def one(sql: str) -> int:
            r = dbm.fetch_one(self.conn, sql)
            return int(r[0]) if r and r[0] is not None else 0

        return {
            "posts": one("SELECT COUNT(*) FROM post"),
            "comments": one("SELECT COUNT(*) FROM comment"),
            "users": one("SELECT COUNT(*) FROM user"),
            "followings": one("SELECT COUNT(*) FROM follow"),
            "followers": one("SELECT COUNT(*) FROM follow"),
        }

    def chain_dimension_share(self, tick: int) -> List[Dict[str, Any]]:
        """某个 tick 上各维度被写出来的比例——用来暴露"模型没按契约输出"。

        这个指标成立的前提是**系统不代模型补值**：只有如实记录每个节点的
        `present` 与 `inherited`，才能区分"模型真的没有发生变化"与
        "解析失败后被兜底填充"。缺节点会立刻在实时页上显形。
        """
        return [
            dict(r) for r in dbm.fetch_all(
                self.conn,
                """SELECT dim, dim_label, node_key, node_label,
                          SUM(present) AS present_n, COUNT(*) AS total_n
                   FROM agent_chain_step WHERE tick=?
                   GROUP BY dim, dim_label, node_key, node_label
                   ORDER BY dim, node_key""",
                (int(tick),),
            )
        ]

    def tool_distribution(self, tick: int) -> List[Dict[str, Any]]:
        return [
            dict(r) for r in dbm.fetch_all(
                self.conn,
                """SELECT tool_name, tool_cn, COUNT(*) AS n, SUM(success) AS ok_n
                   FROM agent_tool_call WHERE tick=?
                   GROUP BY tool_name, tool_cn ORDER BY n DESC""",
                (int(tick),),
            )
        ]


# ======================================================================
# 序列化辅助
# ======================================================================


def _dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None


def _call_to_json(call: Any) -> Dict[str, Any]:
    if hasattr(call, "to_row"):
        row = call.to_row()
    else:
        row = dict(call)
    return {
        "order": row.get("call_order"),
        "tool": row.get("tool_name"),
        "tool_cn": row.get("tool_cn"),
        "args": row.get("arguments_json"),
        "result": row.get("result_json"),
        "success": bool(row.get("success")),
        "error": row.get("error_text"),
    }
