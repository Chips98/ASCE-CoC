"""社交平台核心：内容、互动、推荐、可见性。

设计取舍
--------
1. **显式方法调用，不做反射分发**。动作名与参数类型在调用点写死，执行结果
   直接返回给调用方。这样避免依赖"方法名必须等于某个枚举值""第 N 个参数必须
   叫某个名字"之类的隐式契约——那类约定一旦某处改动就会静默错位，
   而且很难在测试里被发现。
2. **推荐全程用显式 user_id**。不把推荐表读成二维矩阵再按行号反推用户：
   行号与 user_id 的对应关系一旦出现偏移（例如中间插入了控制账号），
   就会把 A 的推荐算到 B 头上，而且不会报错。显式传 ID 列表则没有这个假设。
3. **写动作幂等**。点赞、关注等靠 UNIQUE 约束 + `INSERT OR IGNORE` 保证重复
   调用不产生副作用，因此模型在同一轮多次尝试同一工具是安全的。
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import db as dbm
from .clock import SimClock
from .recsys import (
    DEFAULT_TOPIC_KEYWORDS,
    RecsysInputs,
    RecsysWeights,
    recommend,
    fill_cold_start,
)

logger = logging.getLogger("asce.platform")


class Platform:
    """社交平台。持有唯一一条数据库连接。"""

    def __init__(
        self,
        db_path: str,
        clock: SimClock,
        topic: str = "cut",
        recsys_weights: RecsysWeights | None = None,
        topic_keywords: Dict[str, List[str]] | None = None,
        max_rec_post_len: int = 10,
        refresh_post_count: int = 6,
        max_visible_comments: int = 3,
        allow_self_rating: bool = False,
        seed: int = 42,
    ) -> None:
        self.db_path = db_path
        self.conn = dbm.connect(db_path)
        dbm.create_all(self.conn)
        self.clock = clock
        self.topic = topic
        self.weights = recsys_weights or RecsysWeights()
        self.topic_keywords = topic_keywords or DEFAULT_TOPIC_KEYWORDS
        self.max_rec_post_len = int(max_rec_post_len)
        self.refresh_post_count = int(refresh_post_count)
        self.max_visible_comments = int(max_visible_comments)
        self.allow_self_rating = bool(allow_self_rating)
        self.rng = random.Random(seed)

        self.tick = 0
        # 内存态：本轮每个用户已经看到过的帖子（由 refresh 累积），
        # 用于推荐时施加"已看过惩罚"而无需每轮查库
        self._seen_this_tick: Dict[int, Set[int]] = {}

    # ==================================================================
    # 生命周期
    # ==================================================================
    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def set_tick(self, tick: int) -> None:
        self.tick = int(tick)
        self.clock.set_tick(tick)
        self._seen_this_tick.clear()

    # ==================================================================
    # 用户
    # ==================================================================
    def register_user(self, user_id: int, profile: Dict[str, Any], step: int = 0) -> None:
        """把画像写进 user 表。profile 已由数据清洗脚本去标签化。"""
        now = self.clock.stamp()
        topics = profile.get("interested_topics") or []
        if isinstance(topics, str):
            topics = [topics]
        dbm.execute(
            self.conn,
            """INSERT INTO user(user_id,user_name,name,bio,persona,age,gender,mbti,country,
                               profession,economic_status,education,interested_topics,
                               influence_score,is_control,is_real,num_followings,num_followers,
                               created_at,step_number)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0,?,?)
               ON CONFLICT(user_id) DO UPDATE SET
                 user_name=excluded.user_name, name=excluded.name, bio=excluded.bio,
                 persona=excluded.persona, age=excluded.age, gender=excluded.gender,
                 mbti=excluded.mbti, country=excluded.country, profession=excluded.profession,
                 economic_status=excluded.economic_status, education=excluded.education,
                 interested_topics=excluded.interested_topics,
                 influence_score=excluded.influence_score,
                 is_control=excluded.is_control, is_real=excluded.is_real""",
            (
                int(user_id),
                profile.get("username") or profile.get("display_name") or f"用户{user_id}",
                profile.get("display_name") or profile.get("username") or f"用户{user_id}",
                profile.get("bio") or "",
                profile.get("persona") or "",
                profile.get("age") or "",
                profile.get("gender") or "",
                profile.get("mbti") or "",
                profile.get("country") or "",
                profile.get("profession") or "",
                profile.get("economic_status") or "",
                profile.get("education") or "",
                json.dumps(topics, ensure_ascii=False),
                float((profile.get("influence_metrics") or {}).get("influence_score") or 0.0),
                1 if profile.get("is_control") else 0,
                1 if profile.get("is_real") else 0,
                now,
                int(step),
            ),
        )

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        row = dbm.fetch_one(self.conn, "SELECT * FROM user WHERE user_id=?", (int(user_id),))
        return dict(row) if row else None

    def get_users(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in dbm.fetch_all(self.conn, "SELECT * FROM user ORDER BY user_id")]

    def user_display(self, user_id: int) -> str:
        u = self.get_user(user_id)
        if not u:
            return f"用户{user_id}"
        return u.get("name") or u.get("user_name") or f"用户{user_id}"

    # ==================================================================
    # 发帖 / 评论
    # ==================================================================
    def create_post(
        self,
        user_id: int,
        content: str,
        is_seed: bool = False,
        source_post_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not (content or "").strip():
            return {"success": False, "error": "帖子正文不能为空"}
        step = self.tick if step is None else int(step)
        pid = dbm.execute(
            self.conn,
            """INSERT INTO post(user_id,content,topic,created_at,step_number,num_likes,
                               num_dislikes,num_shares,num_comments,is_seed,source_post_id,
                               conversation_id,extra_json)
               VALUES(?,?,?,?,?,0,0,0,0,?,?,?,?)""",
            (
                int(user_id), content.strip(), self.topic, self.clock.stamp(), step,
                1 if is_seed else 0, source_post_id, conversation_id,
                json.dumps(extra, ensure_ascii=False) if extra else None,
            ),
        )
        self._trace(user_id, "create_post", {"post_id": pid}, step)
        return {"success": True, "post_id": pid}

    def create_comment(
        self,
        user_id: int,
        post_id: int,
        content: str,
        parent_comment_id: Optional[int] = None,
        is_guidance: bool = False,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not (content or "").strip():
            return {"success": False, "error": "评论正文不能为空"}
        post = dbm.fetch_one(self.conn, "SELECT post_id FROM post WHERE post_id=?", (int(post_id),))
        if post is None:
            return {"success": False, "error": f"帖子 {post_id} 不存在"}
        step = self.tick if step is None else int(step)

        if parent_comment_id is not None:
            parent = dbm.fetch_one(
                self.conn, "SELECT comment_id, post_id FROM comment WHERE comment_id=?",
                (int(parent_comment_id),),
            )
            if parent is None:
                return {"success": False, "error": f"评论 {parent_comment_id} 不存在"}
            if int(parent["post_id"]) != int(post_id):
                # 以父评论所属帖子为准，避免评论挂到错误的帖子下
                post_id = int(parent["post_id"])

        cid = dbm.execute(
            self.conn,
            """INSERT INTO comment(post_id,user_id,parent_comment_id,content,created_at,
                                   step_number,num_likes,num_dislikes,num_replies,is_guidance,is_seed)
               VALUES(?,?,?,?,?,?,0,0,0,?,0)""",
            (int(post_id), int(user_id), parent_comment_id, content.strip(),
             self.clock.stamp(), step, 1 if is_guidance else 0),
        )
        dbm.execute(self.conn, "UPDATE post SET num_comments=num_comments+1 WHERE post_id=?", (int(post_id),))
        if parent_comment_id is not None:
            dbm.execute(
                self.conn, "UPDATE comment SET num_replies=num_replies+1 WHERE comment_id=?",
                (int(parent_comment_id),),
            )
        action = "reply_to_comment" if parent_comment_id is not None else "create_comment"
        self._trace(user_id, action, {"post_id": int(post_id), "comment_id": cid}, step)
        return {"success": True, "comment_id": cid, "post_id": int(post_id)}

    def share_comment(self, user_id: int, comment_id: int, step: Optional[int] = None) -> Dict[str, Any]:
        c = dbm.fetch_one(self.conn, "SELECT comment_id FROM comment WHERE comment_id=?", (int(comment_id),))
        if c is None:
            return {"success": False, "error": f"评论 {comment_id} 不存在"}
        step = self.tick if step is None else int(step)
        dbm.execute(
            self.conn,
            "INSERT INTO comment_share(user_id,comment_id,created_at,step_number) VALUES(?,?,?,?)",
            (int(user_id), int(comment_id), self.clock.stamp(), step),
        )
        dbm.execute(
            self.conn, "UPDATE comment SET num_shares=num_shares+1 WHERE comment_id=?",
            (int(comment_id),),
        )
        self._trace(user_id, "share_comment", {"comment_id": int(comment_id)}, step)
        return {"success": True, "comment_id": int(comment_id)}

    # ==================================================================
    # 互动（幂等）
    # ==================================================================
    def like_post(self, user_id: int, post_id: int, step: Optional[int] = None) -> Dict[str, Any]:
        return self._react_post(user_id, post_id, "like", step)

    def dislike_post(self, user_id: int, post_id: int, step: Optional[int] = None) -> Dict[str, Any]:
        return self._react_post(user_id, post_id, "dislike", step)

    def _react_post(self, user_id: int, post_id: int, kind: str, step: Optional[int]) -> Dict[str, Any]:
        p = dbm.fetch_one(
            self.conn, "SELECT post_id,user_id,num_likes,num_dislikes FROM post WHERE post_id=?",
            (int(post_id),),
        )
        if p is None:
            return {"success": False, "error": f"帖子 {post_id} 不存在"}
        if not self.allow_self_rating and int(p["user_id"]) == int(user_id):
            return {"success": False, "error": "不能给自己的帖子评分"}
        step = self.tick if step is None else int(step)
        table = "post_like" if kind == "like" else "post_dislike"
        col = "num_likes" if kind == "like" else "num_dislikes"
        cur = self.conn.cursor()
        cur.execute(
            f"INSERT OR IGNORE INTO {table}(user_id,post_id,created_at,step_number) VALUES(?,?,?,?)",
            (int(user_id), int(post_id), self.clock.stamp(), step),
        )
        inserted = cur.rowcount > 0
        cur.close()
        if inserted:
            dbm.execute(self.conn, f"UPDATE post SET {col}={col}+1 WHERE post_id=?", (int(post_id),))
        self._trace(user_id, f"{kind}_post", {"post_id": int(post_id), "new": inserted}, step)
        return {
            "success": True,
            "post_id": int(post_id),
            "already": not inserted,
            "num_likes": int(p["num_likes"]) + (1 if inserted and kind == "like" else 0),
            "num_dislikes": int(p["num_dislikes"]) + (1 if inserted and kind == "dislike" else 0),
        }

    def like_comment(self, user_id: int, comment_id: int, step: Optional[int] = None) -> Dict[str, Any]:
        return self._react_comment(user_id, comment_id, "like", step)

    def dislike_comment(self, user_id: int, comment_id: int, step: Optional[int] = None) -> Dict[str, Any]:
        return self._react_comment(user_id, comment_id, "dislike", step)

    def _react_comment(self, user_id: int, comment_id: int, kind: str, step: Optional[int]) -> Dict[str, Any]:
        c = dbm.fetch_one(
            self.conn, "SELECT comment_id,user_id,num_likes,num_dislikes FROM comment WHERE comment_id=?",
            (int(comment_id),),
        )
        if c is None:
            return {"success": False, "error": f"评论 {comment_id} 不存在"}
        if not self.allow_self_rating and int(c["user_id"]) == int(user_id):
            return {"success": False, "error": "不能给自己的评论评分"}
        step = self.tick if step is None else int(step)
        col = "num_likes" if kind == "like" else "num_dislikes"
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO comment_reaction(user_id,comment_id,kind,created_at,step_number) "
            "VALUES(?,?,?,?,?)",
            (int(user_id), int(comment_id), kind, self.clock.stamp(), step),
        )
        inserted = cur.rowcount > 0
        cur.close()
        if inserted:
            dbm.execute(self.conn, f"UPDATE comment SET {col}={col}+1 WHERE comment_id=?", (int(comment_id),))
        self._trace(user_id, f"{kind}_comment", {"comment_id": int(comment_id), "new": inserted}, step)
        return {"success": True, "comment_id": int(comment_id), "already": not inserted}

    def follow(self, user_id: int, followee_id: int, step: Optional[int] = None) -> Dict[str, Any]:
        if int(user_id) == int(followee_id):
            return {"success": False, "error": "不能关注自己"}
        if dbm.fetch_one(self.conn, "SELECT user_id FROM user WHERE user_id=?", (int(followee_id),)) is None:
            return {"success": False, "error": f"用户 {followee_id} 不存在"}
        step = self.tick if step is None else int(step)
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO follow(follower_id,followee_id,created_at,step_number) VALUES(?,?,?,?)",
            (int(user_id), int(followee_id), self.clock.stamp(), step),
        )
        inserted = cur.rowcount > 0
        cur.close()
        if inserted:
            dbm.execute(self.conn, "UPDATE user SET num_followings=num_followings+1 WHERE user_id=?", (int(user_id),))
            dbm.execute(self.conn, "UPDATE user SET num_followers=num_followers+1 WHERE user_id=?", (int(followee_id),))
        self._trace(user_id, "follow", {"followee_id": int(followee_id), "new": inserted}, step)
        return {"success": True, "followee_id": int(followee_id), "already": not inserted}

    def unfollow(self, user_id: int, followee_id: int, step: Optional[int] = None) -> Dict[str, Any]:
        step = self.tick if step is None else int(step)
        cur = self.conn.cursor()
        cur.execute(
            "DELETE FROM follow WHERE follower_id=? AND followee_id=?",
            (int(user_id), int(followee_id)),
        )
        removed = cur.rowcount > 0
        cur.close()
        if removed:
            dbm.execute(self.conn, "UPDATE user SET num_followings=MAX(0,num_followings-1) WHERE user_id=?", (int(user_id),))
            dbm.execute(self.conn, "UPDATE user SET num_followers=MAX(0,num_followers-1) WHERE user_id=?", (int(followee_id),))
        self._trace(user_id, "unfollow", {"followee_id": int(followee_id), "removed": removed}, step)
        return {"success": True, "followee_id": int(followee_id), "removed": removed}

    # ==================================================================
    # 只读查询 / 工具动作
    # ==================================================================
    def trend(self, user_id: int, limit: int = 8, step: Optional[int] = None) -> Dict[str, Any]:
        rows = dbm.fetch_all(
            self.conn,
            """SELECT post_id,user_id,content,num_likes,num_dislikes,num_comments
               FROM post ORDER BY (num_likes - num_dislikes) DESC, num_comments DESC LIMIT ?""",
            (int(limit),),
        )
        self._trace(user_id, "trend", {}, step)
        return {"success": True, "posts": [self._brief_post(dict(r)) for r in rows]}

    def search_posts(self, user_id: int, query: str, limit: int = 8, step: Optional[int] = None) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"success": False, "error": "搜索关键词不能为空"}
        rows = dbm.fetch_all(
            self.conn,
            """SELECT post_id,user_id,content,num_likes,num_dislikes,num_comments
               FROM post WHERE content LIKE ? ORDER BY num_likes DESC LIMIT ?""",
            (f"%{q}%", int(limit)),
        )
        self._trace(user_id, "search_posts", {"query": q, "hits": len(rows)}, step)
        return {"success": True, "posts": [self._brief_post(dict(r)) for r in rows]}

    def search_user(self, user_id: int, query: str, limit: int = 8, step: Optional[int] = None) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"success": False, "error": "搜索关键词不能为空"}
        like = f"%{q}%"
        rows = dbm.fetch_all(
            self.conn,
            """SELECT user_id,user_name,name,bio,profession,interested_topics,
                      num_followers,num_followings
               FROM user
               WHERE name LIKE ? OR user_name LIKE ? OR bio LIKE ? OR profession LIKE ?
               ORDER BY num_followers DESC LIMIT ?""",
            (like, like, like, like, int(limit)),
        )
        self._trace(user_id, "search_user", {"query": q, "hits": len(rows)}, step)
        return {"success": True, "users": [dict(r) for r in rows]}

    def get_user_profile(self, user_id: int, target_id: int, step: Optional[int] = None) -> Dict[str, Any]:
        u = self.get_user(target_id)
        self._trace(user_id, "get_user_profile", {"target_id": int(target_id)}, step)
        if not u:
            return {"success": False, "error": f"用户 {target_id} 不存在"}
        return {
            "success": True,
            "user": {
                "user_id": u["user_id"],
                "name": u.get("name"),
                "bio": u.get("bio"),
                "profession": u.get("profession"),
                "country": u.get("country"),
                "interested_topics": _loads_json(u.get("interested_topics"), []),
                "num_followers": u.get("num_followers"),
                "num_followings": u.get("num_followings"),
            },
        }

    def get_post_detail(self, user_id: int, post_id: int, step: Optional[int] = None) -> Dict[str, Any]:
        p = dbm.fetch_one(self.conn, "SELECT * FROM post WHERE post_id=?", (int(post_id),))
        self._trace(user_id, "get_post_detail", {"post_id": int(post_id)}, step)
        if p is None:
            return {"success": False, "error": f"帖子 {post_id} 不存在"}
        post = dict(p)
        post["author_name"] = self.user_display(int(post["user_id"]))
        return {"success": True, "post": post}

    def get_trending_comments(self, user_id: int, limit: int = 5, step: Optional[int] = None) -> Dict[str, Any]:
        rows = dbm.fetch_all(
            self.conn,
            """SELECT c.comment_id, c.post_id, c.user_id, c.content, c.num_likes, c.num_dislikes,
                      p.content AS post_content
               FROM comment c LEFT JOIN post p ON p.post_id = c.post_id
               ORDER BY c.num_likes DESC, c.num_replies DESC LIMIT ?""",
            (int(limit),),
        )
        self._trace(user_id, "get_trending_comments", {"limit": int(limit)}, step)
        out = []
        for r in rows:
            d = dict(r)
            d["author_name"] = self.user_display(int(d["user_id"]))
            out.append(d)
        return {"success": True, "comments": out}

    def do_nothing(self, user_id: int, step: Optional[int] = None) -> Dict[str, Any]:
        self._trace(user_id, "do_nothing", {}, step)
        return {"success": True, "action": "do_nothing"}

    # ==================================================================
    # 可见性：推荐流 + 评论树
    # ==================================================================
    def _comment_tree(self, post_id: int) -> List[Dict[str, Any]]:
        """取某帖的评论树。

        构造规则是**广度受限的完整树**。只挂一层是不够的：回复的回复会因为
        父节点不在字典里而被静默丢弃，评论树就残缺了。这里顶层按热度取
        `max_visible_comments` 条，每条最多带 `max_replies_per_comment` 条回复，
        回复里再带一层，并对总评论数设上限，保证 prompt 体积可预测。
        """
        top_n = self.max_visible_comments
        replies_per = max(2, self.max_visible_comments)
        total_cap = max(10, top_n * 6)

        rows = dbm.fetch_all(
            self.conn,
            """SELECT comment_id,user_id,parent_comment_id,content,num_likes,num_dislikes,
                      num_replies,is_guidance
               FROM comment WHERE post_id=? ORDER BY num_likes DESC, comment_id ASC""",
            (int(post_id),),
        )
        by_parent: Dict[Optional[int], List[Dict[str, Any]]] = {}
        for r in rows:
            d = dict(r)
            d["author_name"] = self.user_display(int(d["user_id"]))
            d["replies"] = []
            by_parent.setdefault(d["parent_comment_id"], []).append(d)

        used = 0
        out: List[Dict[str, Any]] = []

        def attach(node: Dict[str, Any], depth: int) -> None:
            nonlocal used
            if used >= total_cap:
                return
            kids = by_parent.get(node["comment_id"], [])
            for k in kids[:replies_per]:
                if used >= total_cap:
                    return
                used += 1
                k["depth"] = depth + 1
                node["replies"].append(k)
                if depth + 1 < 3:  # 最多三层，避免长尾把 prompt 撑爆
                    attach(k, depth + 1)

        for top in by_parent.get(None, [])[:top_n]:
            used += 1
            top["depth"] = 0
            attach(top, 0)
            out.append(top)
        return out

    def refresh(self, user_id: int, limit: Optional[int] = None, step: Optional[int] = None) -> Dict[str, Any]:
        """返回该用户本 tick 可见的帖子（含评论树）。

        推荐表由 `update_recs()` 在 tick 开始时统一刷新；本方法只负责
        "从推荐里采样一部分 + 拼上关注流 + 挂评论树"，并把曝光记进内存的
        已看集合，供下一轮推荐的"已看过惩罚"使用。
        """
        step = self.tick if step is None else int(step)
        k = int(limit or self.refresh_post_count)

        rec_rows = dbm.fetch_all(
            self.conn,
            "SELECT post_id FROM rec WHERE user_id=? AND step_number=? ORDER BY rank ASC",
            (int(user_id), step),
        )
        rec_ids = [int(r["post_id"]) for r in rec_rows]

        # 关注流：我关注的人在这一 tick 之前发的帖
        follow_rows = dbm.fetch_all(
            self.conn,
            """SELECT p.post_id FROM post p
               JOIN follow f ON f.followee_id = p.user_id
               WHERE f.follower_id=? AND p.step_number <= ?
               ORDER BY p.num_likes DESC, p.post_id DESC LIMIT 3""",
            (int(user_id), step),
        )
        follow_ids = [int(r["post_id"]) for r in follow_rows]

        picked: List[int] = []
        seen: Set[int] = set()
        # 关注流优先，保证"因为我关注了他所以看到"这条因果链可追溯
        for pid in follow_ids + rec_ids:
            if pid not in seen:
                seen.add(pid)
                picked.append(pid)
        if len(picked) > k:
            # 关注流全留，其余从推荐里随机采样，避免每轮 feed 完全相同
            head = picked[: len(follow_ids)]
            tail = picked[len(follow_ids):]
            need = max(0, k - len(head))
            self.rng.shuffle(tail)
            picked = head + tail[:need]
        if not picked:
            rows = dbm.fetch_all(
                self.conn, "SELECT post_id FROM post ORDER BY post_id DESC LIMIT ?", (k,)
            )
            picked = [int(r["post_id"]) for r in rows]

        self._seen_this_tick.setdefault(int(user_id), set()).update(picked)

        posts: List[Dict[str, Any]] = []
        for pid in picked:
            p = dbm.fetch_one(self.conn, "SELECT * FROM post WHERE post_id=?", (pid,))
            if p is None:
                continue
            d = dict(p)
            d["author_name"] = self.user_display(int(d["user_id"]))
            d["comments"] = self._comment_tree(pid)
            posts.append(d)

        self._trace(user_id, "refresh", {"post_ids": picked}, step)
        return {"success": True, "posts": posts}

    # ==================================================================
    # 推荐
    # ==================================================================
    def update_recs(self, user_ids: Sequence[int], step: Optional[int] = None) -> Dict[int, List[int]]:
        """重算本 tick 全部用户的推荐并写 rec 表，返回 `{user_id: [post_id]}`。"""
        step = self.tick if step is None else int(step)
        user_ids = [int(u) for u in user_ids]
        if not user_ids:
            return {}

        posts = [dict(r) for r in dbm.fetch_all(
            self.conn,
            """SELECT post_id,user_id,content,created_at,num_likes,num_dislikes FROM post""",
        )]
        if not posts:
            return {u: [] for u in user_ids}

        #
        comment_counts = {
            int(r["post_id"]): int(r["n"])
            for r in dbm.fetch_all(
                self.conn, "SELECT post_id, COUNT(*) AS n FROM comment GROUP BY post_id"
            )
        }

        follows: Dict[int, Set[int]] = {}
        for r in dbm.fetch_all(self.conn, "SELECT follower_id, followee_id FROM follow"):
            follows.setdefault(int(r["follower_id"]), set()).add(int(r["followee_id"]))

        # 已看过 = 历史推荐 + 我赞过/踩过/评论过的帖子
        seen: Dict[int, Set[int]] = {}
        for r in dbm.fetch_all(
            self.conn, "SELECT DISTINCT user_id, post_id FROM rec", ()
        ):
            seen.setdefault(int(r["user_id"]), set()).add(int(r["post_id"]))
        for tbl in ("post_like", "post_dislike"):
            for r in dbm.fetch_all(self.conn, f"SELECT user_id, post_id FROM {tbl}"):
                seen.setdefault(int(r["user_id"]), set()).add(int(r["post_id"]))
        for r in dbm.fetch_all(self.conn, "SELECT DISTINCT user_id, post_id FROM comment"):
            if r["post_id"] is not None:
                seen.setdefault(int(r["user_id"]), set()).add(int(r["post_id"]))
        # 本 tick 已经刷过 feed 的也算看过
        for uid, pids in self._seen_this_tick.items():
            seen.setdefault(int(uid), set()).update(pids)

        users: List[Dict[str, Any]] = []
        for uid in user_ids:
            u = dbm.fetch_one(
                self.conn,
                "SELECT user_id,persona,bio,interested_topics FROM user WHERE user_id=?",
                (uid,),
            )
            if u is None:
                continue
            users.append({
                "user_id": uid,
                "interested_topics": _loads_json(u["interested_topics"], []),
                "persona_text": f"{u['persona'] or ''} {u['bio'] or ''}".strip(),
            })

        inputs = RecsysInputs(
            now=self.clock.now(),
            users=users,
            posts=posts,
            follows=follows,
            seen=seen,
            comment_counts=comment_counts,
            keywords=self.topic_keywords,
        )
        recs = recommend(inputs, self.max_rec_post_len, self.weights, self.rng)
        recs = fill_cold_start(
            recs, [u["user_id"] for u in users],
            [int(p["post_id"]) for p in posts],
            self.max_rec_post_len, self.rng,
        )

        dbm.execute(self.conn, "DELETE FROM rec WHERE step_number=?", (step,))
        rows: List[Sequence[Any]] = []
        out: Dict[int, List[int]] = {}
        for uid, items in recs.items():
            ids: List[int] = []
            for rank, (pid, score) in enumerate(items):
                rows.append((int(uid), int(pid), step, float(score), rank))
                ids.append(int(pid))
            out[int(uid)] = ids
        dbm.executemany(
            self.conn,
            "INSERT OR REPLACE INTO rec(user_id,post_id,step_number,score,rank) VALUES(?,?,?,?,?)",
            rows,
        )
        return out

    # ==================================================================
    # 初始关注网络
    # ==================================================================
    def bootstrap_follows(
        self,
        user_ids: Sequence[int],
        k: int = 3,
        rng: random.Random | None = None,
        step: int = 0,
    ) -> int:
        """给每个用户建立 k 条初始关注，优先兴趣相近的人。

        相似度用**兴趣话题的 Jaccard 系数**：按相似度取 top-k，同分用随机键打散，
        因此既有同温层效应，又不会让所有人关注完全相同的对象。
        随机源使用传入的局部 `Random` 实例而不动全局 `random`——
        设置全局种子会污染主流程的激活抽样，使复现结果与配置的种子不一致。
        """
        rnd = rng or self.rng
        users = {int(u) for u in user_ids}
        if len(users) < 2:
            return 0
        meta: Dict[int, Set[str]] = {}
        for uid in users:
            row = dbm.fetch_one(self.conn, "SELECT interested_topics FROM user WHERE user_id=?", (uid,))
            meta[uid] = set(_loads_json(row["interested_topics"], []) if row else [])

        made = 0
        for uid in users:
            mine = meta.get(uid, set())
            scored: List[Tuple[float, float, int]] = []
            for other in users:
                if other == uid:
                    continue
                theirs = meta.get(other, set())
                if mine and theirs:
                    union = mine | theirs
                    jac = len(mine & theirs) / len(union) if union else 0.0
                else:
                    jac = 0.0
                scored.append((jac, rnd.random(), other))
            # 兴趣相似度优先，同分随机打散
            scored.sort(key=lambda x: (-x[0], x[1]))
            picks = [s[2] for s in scored[:k]]
            for target in picks:
                res = self.follow(uid, target, step=step)
                if res.get("success") and not res.get("already"):
                    made += 1
        return made

    # ==================================================================
    # 内部工具
    # ==================================================================
    def _trace(self, user_id: int, action: str, info: Dict[str, Any], step: Optional[int]) -> None:
        dbm.execute(
            self.conn,
            "INSERT INTO trace(user_id,action,info,created_at,step_number) VALUES(?,?,?,?,?)",
            (int(user_id), action, json.dumps(info, ensure_ascii=False),
             self.clock.stamp(), self.tick if step is None else int(step)),
        )

    @staticmethod
    def _brief_post(p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "post_id": p.get("post_id"),
            "user_id": p.get("user_id"),
            "content": (p.get("content") or "")[:120],
            "num_likes": p.get("num_likes"),
            "num_dislikes": p.get("num_dislikes"),
            "num_comments": p.get("num_comments"),
        }

    # ==================================================================
    # 统计（供实时页）
    # ==================================================================
    def tick_stat(self, tick: int) -> Dict[str, Any]:
        def one(sql: str, params: Sequence[Any]) -> int:
            r = dbm.fetch_one(self.conn, sql, params)
            return int(r[0]) if r and r[0] is not None else 0

        return {
            "tick": int(tick),
            "posts": one("SELECT COUNT(*) FROM post WHERE step_number<=?", (tick,)),
            "comments": one("SELECT COUNT(*) FROM comment WHERE step_number<=?", (tick,)),
            "likes": one("SELECT COUNT(*) FROM post_like WHERE step_number<=?", (tick,)),
            "follows": one("SELECT COUNT(*) FROM follow", ()),
            "turns": one("SELECT COUNT(*) FROM agent_turn WHERE tick=?", (tick,)),
            "tool_calls": one("SELECT COUNT(*) FROM agent_tool_call WHERE tick=?", (tick,)),
        }


def _loads_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        # 兼容旧格式：逗号分隔的字符串
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return default
