"""智能体：把推荐流、记忆、认知链、工具执行串成一个 tick 的完整因果链。

一个 tick 的流程
----------------
1. `refresh` 拿到本轮可见内容 → 注册进 ToolExecutor 的可见集合
2. 取定向互动（本 tick 收到的干预消息）
3. 渲染输入上下文（画像 / 话题背景 / 记忆 / 最近动作 / 平台现状 / feed / 互动）
4. **第 1 次 LLM 调用**：输出【情绪/情感/思维/立场】，不给任何工具
5. **第 2 次 LLM 调用**：带着第 1 次原文，输出【观点/意图/一致性】并调用工具
6. agentic loop：把工具结果回灌给模型，允许它继续调用（受总次数上限约束）
7. 落库：agent_turn（宽行）+ agent_chain_step（节点明细）+ agent_tool_call（调用明细）

关键设计
--------
* **推理与决策物理隔离**。用两次调用而不是一次：同一次调用里既要求推理、
  又提供动作工具时，模型会先选动作再倒推理由，那样得到的理由是事后合理化，
  不能作为归因依据。
* **曝光可回溯**。每轮把 feed 的 post_id 列表与每个工具调用都落库，
  而不是让 feed 只存在于 prompt 里——否则事后无法回答"它当时看到了什么"。
* **记忆只回看动作与主旨**。不把上一轮的认知结论喂回来，否则模型会倾向复制
  上轮结论，认知链退化成复读。回看轮数可配。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..platform.core import Platform
from ..tools.actions import TOOL_BY_NAME
from ..tools.executor import ToolCallRecord, ToolExecutor
from ..chain.prompts import (
    ChainContext,
    ParsedChain,
    build_stage1,
    build_stage2,
    parse_chain,
)
from ..chain.spec import STAGE1_KEYS, STAGE2_KEYS, Profile, get_profile
from ..persistence.store import TurnStore

logger = logging.getLogger("asce.agent")


@dataclass
class TickResult:
    """一个 tick 的执行结果，供主循环与实时页消费。"""

    agent_id: int
    tick: int
    is_active: bool
    status: str = "ok"
    error: str = ""

    feed_post_ids: List[int] = field(default_factory=list)
    trending_ids: List[int] = field(default_factory=list)
    interaction_ids: List[int] = field(default_factory=list)

    stage1_text: str = ""
    stage2_text: str = ""
    chain1: Optional[ParsedChain] = None
    chain2: Optional[ParsedChain] = None

    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    primary_action: str = ""
    primary_target: Optional[int] = None
    reason: str = ""

    stage1_ms: int = 0
    stage2_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class SocialAgent:
    """一个仿真智能体。"""

    def __init__(
        self,
        agent_id: int,
        profile: Dict[str, Any],
        platform: Platform,
        llm,  # LLMClient
        store: TurnStore,
        chain_profile: Profile,
        topic: str = "cut",
        topic_background: str = "",
        max_tool_calls: int = 3,
        memory_rounds: int = 3,
        temperature: float = 0.7,
        stage1_max_tokens: int = 4000,
        stage2_max_tokens: int = 4000,
        enable_thinking: bool = True,
        max_tools_per_call: int = 3,
    ) -> None:
        self.agent_id = int(agent_id)
        self.profile = dict(profile or {})
        self.platform = platform
        self.llm = llm
        self.store = store
        self.chain_profile = chain_profile
        self.topic = topic
        self.topic_background = topic_background
        self.max_tool_calls = int(max_tool_calls)
        self.memory_rounds = int(memory_rounds)
        self.temperature = float(temperature)
        self.stage1_max_tokens = int(stage1_max_tokens)
        self.stage2_max_tokens = int(stage2_max_tokens)
        self.enable_thinking = bool(enable_thinking)
        self.max_tools_per_call = int(max_tools_per_call)

        # 运行中状态
        self.tick = 0
        self.is_active = False
        self.last_reason = ""

    # ==================================================================
    # 画像与记忆渲染
    # ==================================================================
    def persona_block(self) -> str:
        p = self.profile
        lines = [f"【你的公开资料】"]
        lines.append(f"- 名称：{p.get('display_name') or p.get('username') or f'用户{self.agent_id}'}")
        bio = (p.get("bio") or "").strip()
        if bio:
            lines.append(f"- 简介：{bio}")

        demo = []
        for label, key in (("性别", "gender"), ("年龄段", "age"), ("MBTI", "mbti"),
                           ("国家", "country"), ("职业", "profession")):
            v = p.get(key)
            if v:
                demo.append(f"{label}={v}")
        if demo:
            lines.append("- 人口学：" + " | ".join(demo))

        extra = []
        for label, key in (("经济状况", "economic_status"), ("受教育水平", "education")):
            v = p.get(key)
            if v:
                extra.append(f"{label}={v}")
        if extra:
            lines.append("- 社会背景：" + " | ".join(extra))

        topics = p.get("interested_topics") or []
        if isinstance(topics, list) and topics:
            lines.append("- 长期关注的话题：" + "、".join(str(t) for t in topics))

        infl = (p.get("influence_metrics") or {}).get("influence_score")
        if infl:
            lines.append(f"- 在平台上的影响力评分：{infl}")

        persona = (p.get("persona") or "").strip()
        if persona:
            lines.append(f"- 角色速写：{persona}")
        return "\n".join(lines)

    def memory_text(self) -> str:
        """最近几轮自己做过什么。

        刻意**只回看动作与理由**，不把上轮的认知链结论喂回来：
        否则模型会趋向于复制上轮结论，认知链会退化成"复读机"。
        """
        rows = self.store.recent_actions(self.agent_id, self.tick, self.memory_rounds)
        if not rows:
            return "（还没有历史记录，这是你第一次参与这个讨论）"
        lines = []
        for r in rows:
            t = r["tick"]
            if not r["is_active"]:
                lines.append(f"- 第 {t} 轮：你没有参与讨论")
                continue
            acts = r.get("action_names") or []
            reason = (r.get("reason") or "").strip()
            gist = reason[:40] + ("…" if len(reason) > 40 else "")
            act_txt = "、".join(acts) if acts else "无动作"
            lines.append(f"- 第 {t} 轮：{act_txt}" + (f"｜当时想法：{gist}" if gist else ""))
        return "\n".join(lines)

    def platform_brief(self) -> str:
        row = self.store.platform_brief()
        if not row:
            return ""
        return (
            f"平台目前共有 {row['posts']} 条帖子、{row['comments']} 条评论；"
            f"参与讨论的用户 {row['users']} 人，你关注了 {row['followings']} 人、"
            f"有 {row['followers']} 个关注者。"
        )

    # ==================================================================
    # 截断检测与重试
    # ==================================================================
    async def _call_stage_with_retry(
        self,
        msgs: List[Dict[str, Any]],
        stage_keys,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """调一次模型并解析；若被长度截断且没解析出节点，就用更强的指令重试一次。

        为什么必须做这件事：开启思考模式后模型会先写内部推理，实测它经常把预算
        花在"英文自我检查"上，`finish_reason=length` 时结构化输出**一个字都没写**。
        这种情况必须显式重试、并如实记录最终缺失，而不是静默补一个默认值——
        否则分析时分不清"模型真的这么想"和"系统替它填的"。
        """
        total_ms = 0
        attempt_msgs = list(msgs)
        last_resp = None
        last_parsed = None
        for attempt in range(2):
            t0 = time.monotonic()
            resp = await self.llm.chat(
                attempt_msgs,
                tools=tools,
                temperature=self.temperature,
                max_tokens=max_tokens,
                enable_thinking=self.enable_thinking,
            )
            total_ms += int((time.monotonic() - t0) * 1000)
            text = _usable_text(resp)
            parsed = parse_chain(text, self.chain_profile, stage_keys)
            last_resp, last_parsed = resp, parsed

            if not parsed.missing:
                return resp, parsed, attempt_msgs, total_ms
            if resp.finish_reason != "length":
                # 没截断但仍有缺节点：再催一次通常无效，直接返回
                break
            if attempt == 0:
                logger.info(
                    "agent %s tick %s：输出被长度截断（缺 %d 节点），追加格式指令后重试",
                    self.agent_id, self.tick, len(parsed.missing),
                )
                attempt_msgs = attempt_msgs + [
                    {"role": "assistant", "content": text or ""},
                    {"role": "user", "content": (
                        "你的上一条回复因为太长被截断了，而且**没有按要求写出带节点名的结构化内容**。"
                        "请立刻重新输出：不要写任何思考过程、不要自我检查、不要英文，"
                        "直接从维度标题开始，按下面格式逐行给出，每条 1 句：\n"
                        + self._format_hint(stage_keys)
                    )},
                ]
        assert last_resp is not None and last_parsed is not None
        return last_resp, last_parsed, attempt_msgs, total_ms

    def _format_hint(self, stage_keys) -> str:
        from ..chain.spec import active_nodes
        lines: List[str] = []
        for dim, nodes in active_nodes(self.chain_profile, stage_keys):
            lines.append(f"### {dim.label}")
            for n in nodes:
                lines.append(f"{n.label}：<一句话>")
        return "\n".join(lines)

    # ==================================================================
    # 主流程
    # ==================================================================
    async def run_tick(self, tick: int, interactions: Optional[List[Dict[str, Any]]] = None) -> TickResult:
        self.tick = int(tick)
        self.is_active = True
        interactions = interactions or []
        res = TickResult(agent_id=self.agent_id, tick=self.tick, is_active=True)

        executor = ToolExecutor(self.platform, self.agent_id)

        # ---- 1. 拿本轮 feed ----
        try:
            feed = self.platform.refresh(self.agent_id)
        except Exception as exc:
            logger.exception("agent %s refresh 失败", self.agent_id)
            res.status = "failed"
            res.error = f"refresh 失败: {exc}"
            self._persist(res, executor, None, None)
            return res

        posts = feed.get("posts") or []
        executor.register_feed(posts)
        res.feed_post_ids = [int(p["post_id"]) for p in posts]

        # ---- 2. 热门评论（作为"平台现状"的一部分）----
        trending: List[Dict[str, Any]] = []
        try:
            tr = self.platform.get_trending_comments(self.agent_id, limit=3)
            trending = tr.get("comments") or []
            executor.register_trending(trending)
            res.trending_ids = [int(c["comment_id"]) for c in trending]
        except Exception:
            logger.debug("agent %s 取热门评论失败", self.agent_id, exc_info=True)

        # ---- 3. 定向互动 ----
        for it in interactions:
            iid = it.get("interaction_id")
            if iid is not None:
                res.interaction_ids.append(int(iid))
        for it in interactions:
            if it.get("post_id") is not None:
                executor.visible_posts.add(int(it["post_id"]))
            if it.get("from_id") is not None:
                executor.visible_users.add(int(it["from_id"]))

        # ---- 4. 组装上下文 ----
        ctx = ChainContext(
            agent_id=self.agent_id,
            tick=self.tick,
            persona_text=self.persona_block(),
            feed_text=render_feed(posts, trending, self.platform),
            interactions=interactions,
            memory_text=self.memory_text(),
            recent_actions_text="",
            topic_background=self.topic_background,
            platform_brief=self.platform_brief(),
        )

        # ---- 5. 第 1 次调用：感受与自我（不给工具）----
        try:
            msgs1 = build_stage1(ctx, self.chain_profile, topic=self.topic)
            resp1, parsed1, msgs1, extra_ms = await self._call_stage_with_retry(
                msgs1, STAGE1_KEYS, self.stage1_max_tokens, tools=None
            )
            res.stage1_ms = extra_ms
            res.prompt_tokens += resp1.prompt_tokens
            res.completion_tokens += resp1.completion_tokens
            res.stage1_text = _usable_text(resp1)
            res.chain1 = parsed1
        except Exception as exc:
            logger.exception("agent %s 第 1 阶段失败", self.agent_id)
            res.status = "failed"
            res.error = f"第1阶段失败: {exc}"
            self._persist(res, executor, None, None)
            return res

        # ---- 6. 第 2 次调用 + agentic tool loop ----
        from ..tools.actions import openai_tool_schemas, tool_catalog_text

        tools = openai_tool_schemas()
        catalog = tool_catalog_text()
        stage_label2 = "、".join(d.label for d, _ in __import__(
            "sim.chain.spec", fromlist=["active_nodes"]).active_nodes(self.chain_profile, STAGE2_KEYS))

        msgs = build_stage2(
            ctx,
            self.chain_profile,
            stage1_output=res.chain1.stage_text() or res.stage1_text,
            tool_schemas=tools,
            tool_catalog=catalog,
            topic=self.topic,
        )

        loop = 0
        stage2_texts: List[str] = []
        try:
            while loop < self.max_tool_calls:
                loop += 1
                is_first = loop == 1
                t0 = time.monotonic()
                resp2 = await self.llm.chat(
                    msgs,
                    tools=tools,
                    temperature=self.temperature,
                    max_tokens=self.stage2_max_tokens,
                    enable_thinking=self.enable_thinking,
                )
                res.stage2_ms += int((time.monotonic() - t0) * 1000)
                res.prompt_tokens += resp2.prompt_tokens
                res.completion_tokens += resp2.completion_tokens

                text = _usable_text(resp2)
                if text:
                    stage2_texts.append(text)
                if is_first:
                    res.stage2_text = text
                    res.chain2 = parse_chain(text, self.chain_profile, STAGE2_KEYS)

                if not resp2.tool_calls:
                    # 模型只写了文字、没调工具：追加一次明确的催促，而不是静默补
                    # do_nothing。静默补会让"模型真的什么都不想做"和"系统替它填了
                    # do_nothing"无法区分，是行为统计失真的主要来源。
                    if loop < self.max_tool_calls:
                        msgs.append({"role": "assistant", "content": text or ""})
                        msgs.append({
                            "role": "user",
                            "content": (
                                "你刚才只写了思考，但没有调用任何工具。请现在调用工具执行你的行动。"
                                "如果你确实不想做任何互动，请调用 do_nothing。"
                            ),
                        })
                        continue
                    break

                # 执行本轮全部 tool call（一次响应里可能并行多个）
                assistant_calls = []
                for tc in resp2.tool_calls[: self.max_tools_per_call]:
                    result = executor.execute(tc["name"], tc.get("arguments") or {})
                    assistant_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                        },
                    })
                    msgs.append({"role": "assistant", "content": None, "tool_calls": assistant_calls[-1:]})
                    msgs.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False)[:1500],
                    })

                # 已经有写动作就收工，避免模型在无意义地继续调用只读工具
                if executor.primary_action() is not None:
                    break
        except Exception as exc:
            logger.exception("agent %s 第 2 阶段失败", self.agent_id)
            res.status = "partial"
            res.error = f"第2阶段失败: {exc}"

        res.stage2_text = res.stage2_text or ("\n\n".join(stage2_texts) if stage2_texts else "")
        if res.chain2 is None:
            res.chain2 = parse_chain(res.stage2_text, self.chain_profile, STAGE2_KEYS)

        # ---- 7. 兜底：一次成功的工具调用都没有 ----
        if not any(r.success for r in executor.calls):
            executor.execute("do_nothing", {})
            res.status = "fallback" if res.status == "ok" else res.status

        res.tool_calls = executor.calls
        prim = executor.primary_action()
        if prim is not None:
            res.primary_action = prim.name
            res.primary_target = executor.target_of(prim)
        res.reason = self._derive_reason(res)
        self.last_reason = res.reason

        self._persist(res, executor, res.chain1, res.chain2)
        return res

    def run_tick_skipped(self, tick: int) -> TickResult:
        """未被激活的轮次也要留痕，否则"活跃/非活跃"的对比分析没有基线。"""
        self.tick = int(tick)
        self.is_active = False
        res = TickResult(agent_id=self.agent_id, tick=self.tick, is_active=False, status="skipped")
        self._persist(res, ToolExecutor(self.platform, self.agent_id), None, None)
        return res

    # ==================================================================
    # 理由推导
    # ==================================================================
    def _derive_reason(self, res: TickResult) -> str:
        """取最贴近"为什么这么做"的一段作为本轮行为理由。

        取值优先级（全部来自已有记录，不做任何再生成）：
          1. **意图维度的第一个节点**——它直接回答"我为什么想做这件事"。
          2. 一致性收口——说明"从情绪一路怎么走到动作"。
          3. 观点维度的最后一个节点。
          4. 立场定位（兜底）。

        刻意不取意图维度的**最后一个**节点：多节点档位下那是"感知行为控制"，
        内容形如"我有能力完成这个操作"，是能力自述而非行为理由。
        """
        if res.chain2 is not None:
            steps = res.chain2.steps
            for dim_key in ("intention", "integration", "opinion", "stance"):
                texts = [s.text for s in steps if s.dim == dim_key and s.present]
                if not texts:
                    continue
                # 意图取第一个（态度），其余维度取最后一个（结论）
                return _first_sentence(texts[0] if dim_key == "intention" else texts[-1], 140)
        if res.stage2_text:
            for line in res.stage2_text.strip().split("\n"):
                if line.strip() and not line.strip().startswith("#"):
                    return _first_sentence(line.strip(), 140)
        return ""

    # ==================================================================
    # 落库
    # ==================================================================
    def _persist(
        self,
        res: TickResult,
        executor: ToolExecutor,
        chain1: Optional[ParsedChain],
        chain2: Optional[ParsedChain],
    ) -> None:
        try:
            thought_parts = []
            if res.stage1_text:
                thought_parts.append("【第一阶段｜感受与自我】\n" + res.stage1_text)
            if res.stage2_text:
                thought_parts.append("【第二阶段｜判断与行动】\n" + res.stage2_text)

            self.store.save_turn(
                agent_id=res.agent_id,
                tick=res.tick,
                is_active=res.is_active,
                status=res.status,
                error=res.error,
                feed_post_ids=res.feed_post_ids,
                trending_ids=res.trending_ids,
                interaction_ids=res.interaction_ids,
                thought_text="\n\n".join(thought_parts),
                stage1_text=res.stage1_text,
                stage2_text=res.stage2_text,
                chain_steps={"stage1": chain1.to_json() if chain1 else None,
                             "stage2": chain2.to_json() if chain2 else None},
                chains=[(1, chain1), (2, chain2)],
                tool_calls=executor.calls,
                primary_action=res.primary_action,
                primary_target=res.primary_target,
                reason=res.reason,
                profile=self.chain_profile.key,
                stage1_ms=res.stage1_ms,
                stage2_ms=res.stage2_ms,
                prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                model=getattr(self.llm, "model", ""),
            )
        except Exception:
            logger.exception("agent %s tick %s 落库失败", res.agent_id, res.tick)


def _usable_text(resp) -> str:
    """取出可用于解析的正文。

    **不要**在正文为空时退回 reasoning：`_split_think` 已经区分了三种情况——
    带闭合标签时 content 就是正文；未闭合（被长度截断）时 content 为空而
    reasoning 是整段内部推理。此时若退回 reasoning，就会把"英文元评论式推理"
    当成结构化输出解析，实测污染了 249/445 个 turn。正文为空就该走截断重试。
    """
    return (resp.content or "").strip()


def _first_sentence(text: str, limit: int = 140) -> str:
    """取首句并限长。理由字段只是索引，完整推理在认知链里。"""
    t = (text or "").strip()
    for sep in ("。", "；", ";", "\n"):
        i = t.find(sep)
        if 0 < i <= limit:
            return t[: i + 1]
    return t[:limit]


# ======================================================================
# feed 渲染
# ======================================================================


def render_feed(posts: Sequence[Dict[str, Any]], trending: Sequence[Dict[str, Any]],
                platform: Platform) -> str:
    """把推荐流渲染成给模型看的文本。

    用紧凑的中文可读格式，而不是直接 `json.dumps` 整个结构：后者会把嵌套的
    评论树原样序列化，一个 feed 就可能上千 token，而 `created_at`、
    `source_post_id` 这类字段对模型判断毫无用处。这里只保留模型真正需要的信息。
    """
    if not posts:
        return "（本轮没有刷到任何内容）"

    out: List[str] = []
    for p in posts:
        out.append(
            f"【帖子 {p['post_id']}】作者：{p.get('author_name') or platform.user_display(int(p['user_id']))}"
            f"（ID {p['user_id']}）"
        )
        out.append(f"  内容：{(p.get('content') or '').strip()}")
        out.append(
            f"  热度：赞 {p.get('num_likes', 0)} · 踩 {p.get('num_dislikes', 0)} · "
            f"评论 {p.get('num_comments', 0)}"
        )
        comments = p.get("comments") or []
        if comments:
            for c in comments:
                _render_comment(c, out)
        else:
            out.append("  （还没有人评论）")
        out.append("")

    if trending:
        out.append("【平台当前的热门评论】")
        for c in trending:
            out.append(
                f"- 评论 {c['comment_id']}（作者 {c.get('author_name') or c.get('user_id')}，"
                f"赞 {c.get('num_likes', 0)}）在帖子 {c.get('post_id')} 下：{(c.get('content') or '').strip()}"
            )
        out.append("")
    return "\n".join(out).rstrip()


def _render_comment(c: Dict[str, Any], out: List[str], depth: int = 0) -> None:
    pad = "  " + "  " * depth
    marker = "└ " if depth else "└ "
    out.append(
        f"{pad}{marker}评论 {c['comment_id']}（{c.get('author_name') or c.get('user_id')}，"
        f"赞 {c.get('num_likes', 0)}）：{(c.get('content') or '').strip()}"
    )
    if depth < 2:
        for r in c.get("replies") or []:
            _render_comment(r, out, depth + 1)
