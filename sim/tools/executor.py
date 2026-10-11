"""工具执行器：把模型的 tool call 映射到平台方法，并做上下文校验。

关键设计：**动作只能作用于 agent 本轮确实看见过的对象**。

实现这条约束时有两个要点：白名单必须在**动作生效前**校验，且被拒绝的调用
必须留下痕迹。若采用"先收集可见 ID、再过滤参数"的做法，被过滤掉的参数会
静默消失——于是流水里记录了动作，而平台状态并没有改变，行为统计与仿真状态
就对不上了。本实现的做法是：

* 非法 ID **不静默丢弃**，而是返回结构化错误 `{"success": false, "error": ...}`，
  模型能在同一个 agentic loop 里读到错误并改正；
* 可见集合**动态扩展**：调用 `get_post_detail` / `search_posts` 之后，
  新查到的 ID 立即变为可见，于是"先看详情、再评论"这条自然行为链能走通；
* 每个工具按 `max_calls_per_tick` 限次，防止模型刷调用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..tools.actions import TOOL_BY_NAME, coerce_arguments, validate_arguments
from ..platform.core import Platform

logger = logging.getLogger("asce.tools")


@dataclass
class ToolCallRecord:
    """一次工具调用的完整记录，用于落库与实时页展示。"""

    order: int
    name: str
    cn_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    error: str = ""

    def to_row(self) -> Dict[str, Any]:
        return {
            "call_order": self.order,
            "tool_name": self.name,
            "tool_cn": self.cn_name,
            "arguments_json": self.arguments,
            "result_json": self.result,
            "success": 1 if self.success else 0,
            "error_text": self.error or None,
        }


def _check_content(name: str, args: Dict[str, Any]) -> str:
    """检查发言类工具的内容是否是"真的发言"，而不是元推理。

    实测问题：模型会把认知链的推理过程（"我所在的经济分析圈层普遍期待理性讨论
    ……这种群体压力会促使我……"）直接当成评论正文发出去。这种内容一旦入库，
    整个仿真就不可信了——帖子下面全是自我分析，没有一句真实社交发言。

    检测策略：元推理有很稳定的用词特征（谈论自己的压力、维度、倾向、圈层期待），
    而真实发言谈的是外部事件。用关键词表拦截，并给出重写指引。
    """
    if name not in ("create_comment", "reply_to_comment", "create_post"):
        return ""
    content = str(args.get("content") or "").strip()
    if not content:
        return ""

    if len(content) < 6:
        return "发言内容太短，不像一句完整的话，请写 1 到 3 句自然的中文。"

    meta_markers = (
        "认知链", "这一维度", "各个维度", "跨维度", "元认知",
        "圈层期待", "群体压力会", "群体期待我", "社会期待",
        "我的确定性", "可及性高", "可及性强", "态度强度",
        "一致性检查", "链条自洽", "自洽性",
        "我会遵循", "我会保持", "促使我", "强化我",
        "我的立场是", "我持", "基于上述", "综上",
        "知觉到", "感知到群体", "规范性压力",
    )
    hits = [m for m in meta_markers if m in content]
    if hits:
        return (
            "这条内容读起来像是在自我分析，而不是在跟别人说话（命中了："
            + "、".join(hits[:3])
            + "）。请改成你在社交媒体上真的会发出去的那句话："
            "直接谈帖子里的具体事件或观点，第一人称，口语化，1 到 3 句。"
        )

    # 元推理往往有"说明文"腔：大量抽象名词 + 没有具体指代
    abstract = sum(content.count(w) for w in ("层面", "角度", "机制", "逻辑性", "倾向性"))
    if abstract >= 3:
        return (
            "这条内容太抽象，没有谈到帖子里的具体内容。"
            "请写一句具体的话，比如指出哪个数字、哪条政策、哪种后果。"
        )
    return ""


class ToolExecutor:
    """按 tick 复用的工具执行器。"""

    def __init__(self, platform: Platform, agent_id: int) -> None:
        self.platform = platform
        self.agent_id = int(agent_id)

        # 本轮可见集合
        self.visible_posts: Set[int] = set()
        self.visible_comments: Set[int] = set()
        self.visible_users: Set[int] = set()

        self.calls: List[ToolCallRecord] = []
        self._counts: Dict[str, int] = {}

        # 被平台创建的对象的 id（帖子/评论），供落库时标注
        self.created_post_id: Optional[int] = None
        self.created_comment_id: Optional[int] = None

    # ------------------------------------------------------------------ 可见性
    def register_feed(self, posts: List[Dict[str, Any]]) -> None:
        """把本轮的推荐流注册为可见。

        评论只注册**展示出来的那部分**（含 replies），与 prompt 里 agent 真正
        读到的内容严格一致——否则 agent 可能对一条它没看到的评论作出反应，
        事后无法解释行为来源。
        """
        for p in posts or []:
            pid = p.get("post_id")
            if pid is None:
                continue
            self.visible_posts.add(int(pid))
            author = p.get("user_id")
            if author is not None:
                self.visible_users.add(int(author))
            for c in p.get("comments") or []:
                self._register_comment(c)

    def register_trending(self, comments: List[Dict[str, Any]]) -> None:
        for c in comments or []:
            cid = c.get("comment_id")
            if cid is not None:
                self.visible_comments.add(int(cid))
            pid = c.get("post_id")
            if pid is not None:
                self.visible_posts.add(int(pid))
            uid = c.get("user_id")
            if uid is not None:
                self.visible_users.add(int(uid))

    def _register_comment(self, c: Dict[str, Any]) -> None:
        cid = c.get("comment_id")
        if cid is not None:
            self.visible_comments.add(int(cid))
        uid = c.get("user_id")
        if uid is not None:
            self.visible_users.add(int(uid))
        for r in c.get("replies") or []:
            self._register_comment(r)

    # ------------------------------------------------------------------ 执行
    def remaining_quota(self, name: str) -> int:
        spec = TOOL_BY_NAME.get(name)
        cap = spec.max_calls_per_tick if spec else 1
        return max(0, cap - self._counts.get(name, 0))

    def execute(self, name: str, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        """执行一次工具调用，永远返回一个 dict（失败也返回 success=False）。"""
        spec = TOOL_BY_NAME.get(name)
        order = len(self.calls) + 1

        if spec is None:
            rec = ToolCallRecord(order, name, name, dict(raw_args or {}),
                                 {"success": False, "error": f"未知工具 {name}"}, False,
                                 f"未知工具 {name}")
            self.calls.append(rec)
            return rec.result

        if self.remaining_quota(name) <= 0:
            msg = f"{spec.cn_name} 本轮已达调用上限（{spec.max_calls_per_tick} 次），请换一个动作或结束"
            rec = ToolCallRecord(order, name, spec.cn_name, dict(raw_args or {}),
                                 {"success": False, "error": msg}, False, msg)
            self.calls.append(rec)
            return rec.result

        args = coerce_arguments(name, dict(raw_args or {}))
        errors = validate_arguments(name, args)
        if errors:
            msg = "；".join(errors)
            rec = ToolCallRecord(order, name, spec.cn_name, args,
                                 {"success": False, "error": msg}, False, msg)
            self.calls.append(rec)
            # 参数错误也算一次调用，避免模型用错误参数刷满配额
            self._counts[name] = self._counts.get(name, 0) + 1
            return rec.result

        # 上下文校验：只能操作看见过的对象
        ctx_err = self._check_context(name, args)
        if ctx_err:
            rec = ToolCallRecord(order, name, spec.cn_name, args,
                                 {"success": False, "error": ctx_err}, False, ctx_err)
            self.calls.append(rec)
            self._counts[name] = self._counts.get(name, 0) + 1
            return rec.result

        # 内容质量校验：防止模型把"元推理"当成发言内容
        content_err = _check_content(name, args)
        if content_err:
            rec = ToolCallRecord(order, name, spec.cn_name, args,
                                 {"success": False, "error": content_err}, False, content_err)
            self.calls.append(rec)
            self._counts[name] = self._counts.get(name, 0) + 1
            return rec.result

        try:
            result = self._dispatch(name, args)
        except Exception as exc:  # 任何平台异常都转成模型可读的错误
            logger.exception("工具 %s 执行异常: %s", name, exc)
            msg = f"执行失败：{exc}"
            rec = ToolCallRecord(order, name, spec.cn_name, args,
                                 {"success": False, "error": msg}, False, msg)
            self.calls.append(rec)
            self._counts[name] = self._counts.get(name, 0) + 1
            return rec.result

        self._counts[name] = self._counts.get(name, 0) + 1
        ok = bool(result.get("success"))
        rec = ToolCallRecord(order, name, spec.cn_name, args, result, ok,
                             "" if ok else str(result.get("error") or ""))
        self.calls.append(rec)
        return result

    # ------------------------------------------------------------------ 上下文校验
    def _check_context(self, name: str, args: Dict[str, Any]) -> str:
        if name in ("like_post", "dislike_post", "create_comment", "get_post_detail"):
            pid = args.get("post_id")
            if pid is not None and int(pid) not in self.visible_posts:
                return (
                    f"帖子 {pid} 不在你本轮看到的内容里。"
                    f"你只能对你刷到的帖子操作；如果需要看更多内容，可以先调用 refresh 或 search_posts。"
                )
        if name in ("like_comment", "dislike_comment", "reply_to_comment", "share_comment"):
            cid = args.get("comment_id")
            if cid is not None and int(cid) not in self.visible_comments:
                return f"评论 {cid} 不在你本轮看到的内容里，无法对它操作。"
        if name == "follow":
            fid = args.get("followee_id")
            if fid is not None and int(fid) not in self.visible_users:
                return f"用户 {fid} 你本轮没有接触过，无法关注。先看看他的帖子或资料。"
        if name == "unfollow":
            fid = args.get("followee_id")
            if fid is not None:
                row = self.platform.conn.execute(
                    "SELECT 1 FROM follow WHERE follower_id=? AND followee_id=?",
                    (self.agent_id, int(fid)),
                ).fetchone()
                if row is None:
                    return f"你并没有关注用户 {fid}，无法取消关注。"
        return ""

    # ------------------------------------------------------------------ 分发
    def _dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        p = self.platform
        uid = self.agent_id

        if name == "like_post":
            return p.like_post(uid, args["post_id"])
        if name == "dislike_post":
            return p.dislike_post(uid, args["post_id"])
        if name == "create_comment":
            r = p.create_comment(uid, args["post_id"], args["content"])
            if r.get("success"):
                self.created_comment_id = int(r["comment_id"])
                self.visible_comments.add(int(r["comment_id"]))
            return r
        if name == "reply_to_comment":
            r = p.create_comment(uid, 0, args["content"], parent_comment_id=args["comment_id"])
            if r.get("success"):
                self.created_comment_id = int(r["comment_id"])
                self.visible_comments.add(int(r["comment_id"]))
                self.visible_posts.add(int(r["post_id"]))
            return r
        if name == "share_comment":
            return p.share_comment(uid, args["comment_id"])
        if name == "like_comment":
            return p.like_comment(uid, args["comment_id"])
        if name == "dislike_comment":
            return p.dislike_comment(uid, args["comment_id"])
        if name == "create_post":
            r = p.create_post(uid, args["content"])
            if r.get("success"):
                self.created_post_id = int(r["post_id"])
                self.visible_posts.add(int(r["post_id"]))
            return r
        if name == "follow":
            return p.follow(uid, args["followee_id"])
        if name == "unfollow":
            return p.unfollow(uid, args["followee_id"])
        if name == "do_nothing":
            return p.do_nothing(uid)
        if name == "refresh":
            r = p.refresh(uid)
            if r.get("success"):
                self.register_feed(r.get("posts") or [])
                # 回给模型的 refresh 结果要瘦身，避免把整棵评论树塞回上下文
                r = {"success": True, "post_ids": [int(x["post_id"]) for x in (r.get("posts") or [])],
                     "note": "信息流已刷新，新帖子已加入你可见的内容"}
            return r
        if name == "trend":
            r = p.trend(uid)
            if r.get("success"):
                for item in r.get("posts") or []:
                    self.visible_posts.add(int(item["post_id"]))
            return r
        if name == "search_posts":
            r = p.search_posts(uid, args["query"])
            if r.get("success"):
                for item in r.get("posts") or []:
                    self.visible_posts.add(int(item["post_id"]))
            return r
        if name == "search_user":
            r = p.search_user(uid, args["query"])
            if r.get("success"):
                for item in r.get("users") or []:
                    self.visible_users.add(int(item["user_id"]))
            return r
        if name == "get_trending_comments":
            r = p.get_trending_comments(uid, args.get("limit") or 5)
            if r.get("success"):
                self.register_trending(r.get("comments") or [])
            return r
        if name == "get_post_detail":
            r = p.get_post_detail(uid, args["post_id"])
            if r.get("success"):
                self.visible_posts.add(int(r["post"]["post_id"]))
                self.visible_users.add(int(r["post"]["user_id"]))
            return r
        if name == "get_user_profile":
            r = p.get_user_profile(uid, args["user_id"])
            if r.get("success"):
                self.visible_users.add(int(args["user_id"]))
            return r

        return {"success": False, "error": f"工具 {name} 还没有实现"}

    # ------------------------------------------------------------------ 汇总
    def primary_action(self) -> Optional[ToolCallRecord]:
        """主动作 = 第一个成功的**写**动作；没有写动作则取第一个成功调用。"""
        for rec in self.calls:
            if rec.success and TOOL_BY_NAME.get(rec.name) and TOOL_BY_NAME[rec.name].kind == "write":
                return rec
        for rec in self.calls:
            if rec.success:
                return rec
        return None

    def target_of(self, rec: ToolCallRecord) -> Optional[int]:
        for key in ("post_id", "comment_id", "followee_id", "user_id"):
            if key in rec.arguments:
                try:
                    return int(rec.arguments[key])
                except (TypeError, ValueError):
                    return None
        return None
