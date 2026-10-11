"""推荐系统。

设计目标
--------
推荐打分要同时满足三个条件：**特征可解释**、**不会静默失效**、
**在几百个智能体规模下仍然快**。对应的做法是：

1. **预取式打分**：一次性把 user / post / follow / seen 全部读进内存，
   打分阶段是纯 Python 计算，不做逐条查询。这是大规模运行下最主要的性能来源。
2. **显式 user_id**：全程传 ID 列表，不用"矩阵行号 + 固定偏移"反推用户——
   行序与 ID 一旦出现偏移，就会把推荐算到别人头上且不会报错。
3. **特征只依赖真实存在的字段**：
   - 热度用净赞数与评论数计算，不依赖任何外部魔数；
   - 时间衰减显式接收仿真时钟，保证真正生效；
   - 个性化用 `interest_match`（兴趣话题 × 帖子正文）与 `persona_match`
     （人设文本 × 帖子正文）。两者都只依赖画像里确实存在的字段，
     不会因为某个标注缺失而整项退化为 0 却不被察觉。
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------
# 文本匹配工具（无第三方依赖，纯关键词加权）
# --------------------------------------------------------------------------

# 中文分词能力有限，这里用"关键词 + 字符二元组重叠"的混合策略：
# 关键词命中给高权重，二元组重叠给低权重，保证不同 agent 的排序有差异，
# 又不至于因为分词失败而全部相同。
_PUNCT = re.compile(
    r"[\s，。！？、；：“”‘’（）《》【】,.!?;:()\[\]<>\"'`~@#$%^&*_+=\-|/\\]+"
)

# 参与重叠度计算的正文长度上限。cut 语料最长 1238 字，若全文参与 bigram 集合运算，
# 100 agent × 2683 帖的启动打分会产生上百万次集合操作。截断到上限后集合大小有界，
# 而社交媒体的兴趣匹配本就由开头几句决定，截断几乎不损失判别力。
_TEXT_CAP = 220


def _bigrams(text: str) -> Set[str]:
    t = _PUNCT.sub("", (text or "")[:_TEXT_CAP])
    if len(t) < 2:
        return {t} if t else set()
    return {t[i : i + 2] for i in range(len(t) - 1)}


def _tokens(text: str) -> Set[str]:
    """粗粒度切词：按标点切段 + 二元组。"""
    text = (text or "")[:_TEXT_CAP]
    parts = [p for p in _PUNCT.split(text) if p]
    out: Set[str] = set(parts)
    out |= _bigrams(text)
    return {t for t in out if len(t) >= 2}


def text_overlap(a: str, b: str) -> float:
    """两个中文短文本的重叠度，输出 [0, 1]。"""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    # 用较小集合归一化：帖子短、画像长，用 min 避免被长文本稀释
    return inter / min(len(ta), len(tb))


# 话题关键词表：把"话题"映射到一组特征词，用于 interest_match。
# 保持为可配置的字典而不是散落在函数里的字面量：接入新话题或新语料时，
# 只需要扩展这里，不必改动打分逻辑。
DEFAULT_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "国际关系": ["中美", "美方", "中方", "关税", "谈判", "外交", "制裁", "出口", "进口", "贸易"],
    "经济": ["经济", "通胀", "价格", "成本", "市场", "企业", "消费者", "GDP", "就业", "产业链"],
    "政治": ["政策", "政府", "国会", "选举", "立法", "白宫", "人大", "监管"],
    "科技": ["芯片", "半导体", "技术", "AI", "人工智能", "研发", "算力", "稀土"],
    "农业": ["农民", "农产品", "大豆", "粮食", "采购", "种植"],
    "军事": ["军费", "军事", "安全", "战略", "冲突", "海峡"],
    "教育": ["教育", "学校", "学生", "大学", "课程", "培养"],
    "环境": ["气候", "碳排", "环保", "能源", "清洁", "绿色"],
    "社会民生": ["民生", "社保", "医疗", "养老", "房价", "收入"],
}


# --------------------------------------------------------------------------
# 打分特征
# --------------------------------------------------------------------------


def hot_score(num_likes: int, num_dislikes: int, num_comments: int = 0) -> float:
    """热度：净赞数取对数，叠加评论数（评论算半权重）。

    用对数压缩长尾：少数帖子可能获得远高于平均的互动量，线性加权会让它们
    长期霸占推荐位，新内容没有出头机会。被踩多于被赞时给一个轻微惩罚，
    但不取负值——否则帖子一旦沉底就再无翻身可能。
    """
    net = (num_likes or 0) - (num_dislikes or 0)
    comments = num_comments or 0
    # log1p 保证 0 分帖子热度为 0 而不是负无穷
    h = math.log1p(max(net, 0)) + 0.5 * math.log1p(max(comments, 0))
    # 被踩多于被赞时给一个轻微惩罚（不为负，避免完全沉底后永无翻身）
    if net < 0:
        h *= 0.5
    return h


def time_decay(created_at: datetime, now: datetime, half_life_hours: float = 12.0) -> float:
    """指数时间衰减，半衰期默认 12 仿真小时。

    `now` 必须由调用方传入**仿真时钟**的当前时间，而不是取真实系统时间：
    帖子的 `created_at` 来自仿真时钟，与真实时间不同源，
    混用会让衰减系数恒为 1、整个特征静默失效。
    """
    if created_at is None or now is None:
        return 1.0
    age_hours = max((now - created_at).total_seconds() / 3600.0, 0.0)
    return float(math.exp(-age_hours / max(half_life_hours, 1e-6)))


def interest_match(post_content: str, interests: Sequence[str], keywords: Dict[str, List[str]] | None = None) -> float:
    """兴趣匹配：agent 的兴趣话题关键词与帖子文本的重合度，输出 [0, 1]。"""
    if not interests:
        return 0.0
    kw = keywords or DEFAULT_TOPIC_KEYWORDS
    content = post_content or ""
    matched_topics = 0
    for topic in interests:
        words = kw.get(topic)
        if words:
            hit = sum(1 for w in words if w in content)
            if hit:
                matched_topics += 1
        elif topic and topic in content:
            matched_topics += 1
    if matched_topics == 0:
        return 0.0
    return min(1.0, matched_topics / max(len(interests), 1))


def persona_match(persona_text: str, post_content: str) -> float:
    """人设匹配：agent 人设文本与帖子正文的重叠度。

    衡量的是"这条内容是否契合我这个人"。它只依赖画像里本来就有的
    人设与职业描述，不需要任何额外标注字段，因此不会因为标注缺失而整项失效。
    """
    return text_overlap(persona_text, post_content)


# --------------------------------------------------------------------------
# 上下文预取
# --------------------------------------------------------------------------


@dataclass
class RecsysInputs:
    """一次推荐计算所需的全部数据（已预取，打分阶段不再查库）。"""

    now: datetime
    users: List[Dict[str, Any]] = field(default_factory=list)          # 只含需要推荐的用户
    posts: List[Dict[str, Any]] = field(default_factory=list)          # 候选帖
    follows: Dict[int, Set[int]] = field(default_factory=dict)         # follower -> {followee}
    seen: Dict[int, Set[int]] = field(default_factory=dict)            # user -> {post_id}
    comment_counts: Dict[int, int] = field(default_factory=dict)       # post_id -> 评论数
    keywords: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class RecsysWeights:
    """推荐权重。总和不必为 1，最终会归一化比较。"""

    hot: float = 0.20
    time: float = 0.15
    follow: float = 0.30
    interest: float = 0.20
    persona: float = 0.10
    random: float = 0.05
    # 已看过的帖子打几折。给 0.35 而不是压到接近 0：语料规模有限时
    # 必须允许一定程度的重复曝光，否则候选池会被迅速耗尽；
    # 但折扣要足够低，让未见过的内容优先出现。
    seen_penalty: float = 0.35
    half_life_hours: float = 12.0


# --------------------------------------------------------------------------
# 主推荐函数
# --------------------------------------------------------------------------


def recommend(
    inputs: RecsysInputs,
    top_k: int,
    weights: RecsysWeights | None = None,
    rng: random.Random | None = None,
) -> Dict[int, List[Tuple[int, float]]]:
    """为每个用户算出 top_k 推荐。

    返回 `{user_id: [(post_id, score), ...]}`，分数已按降序排列。

    性能要点：
    * 帖子侧的"与用户无关"特征（热度、时间衰减、切词结果）**只算一次**，
      所有用户共享；
    * 用户侧特征（follow / interest / persona）逐用户算，但依赖的 follow 集合、
      兴趣词表都是内存查表，没有 SQL；
    * 取 top-k 用 partial selection（`heapq.nlargest`）而不是全排序：
      候选帖上万条时快一个量级。
    """
    import heapq

    w = weights or RecsysWeights()
    rnd = rng or random.Random()
    kw = inputs.keywords or DEFAULT_TOPIC_KEYWORDS

    # ---- 帖子共享特征 ----
    # 帖子侧的 token 集合**只算一次**。朴素写法是在用户循环里对每个
    # (用户, 帖子) 重新切词，100 agent × 2683 帖 = 26.8 万次集合构造，
    # 实测占掉单轮 7.7s 里的绝大部分。预计算后同样的打分只需几十毫秒。
    post_base: Dict[int, float] = {}
    post_author: Dict[int, int] = {}
    post_content: Dict[int, str] = {}
    post_tokens: Dict[int, Set[str]] = {}
    for p in inputs.posts:
        pid = int(p["post_id"])
        post_author[pid] = int(p.get("user_id") or 0)
        content = p.get("content") or ""
        post_content[pid] = content
        post_tokens[pid] = _tokens(content)
        h = hot_score(
            int(p.get("num_likes") or 0),
            int(p.get("num_dislikes") or 0),
            inputs.comment_counts.get(pid, 0),
        )
        t = time_decay(_parse_dt(p.get("created_at")), inputs.now, w.half_life_hours)
        post_base[pid] = w.hot * h + w.time * t

    # 兴趣话题 -> 关键词集合，同样预先展开，避免逐条帖子做关键词 in 判断
    topic_words: Dict[str, List[str]] = {}
    for topic in {t for u in inputs.users for t in (u.get("interested_topics") or [])}:
        topic_words[str(topic)] = list(kw.get(str(topic)) or [])

    # 热度归一化：log 之后的量纲与 [0,1] 的其它特征不一致，压缩到 [0,1]
    if post_base:
        hi = max(post_base.values())
        if hi > 0:
            inv = 1.0 / hi
            for pid in post_base:
                post_base[pid] *= inv

    all_post_ids = list(post_base.keys())
    out: Dict[int, List[Tuple[int, float]]] = {}

    for u in inputs.users:
        uid = int(u["user_id"])
        interests = [str(x) for x in (u.get("interested_topics") or [])]
        persona_tokens = _tokens(u.get("persona_text") or "")
        follows = inputs.follows.get(uid, set())
        seen = inputs.seen.get(uid, set())

        # 用户侧也预先求出兴趣词表，避免在内层循环里查字典
        user_words: List[str] = []
        for topic in interests:
            user_words.extend(topic_words.get(topic, []))
            if not topic_words.get(topic) and topic:
                user_words.append(topic)
        n_interests = max(len(interests), 1)
        p_len = len(persona_tokens)

        scored: List[Tuple[int, float]] = []
        for pid in all_post_ids:
            author = post_author.get(pid, 0)
            if author == uid:
                continue  # 不给自己推荐自己的帖子

            s = post_base.get(pid, 0.0)
            if author in follows:
                s += w.follow

            if user_words:
                content = post_content.get(pid, "")
                hit_topics = sum(
                    1 for topic in interests if any(x in content for x in topic_words.get(topic, ()))
                    or (not topic_words.get(topic) and topic and topic in content)
                )
                if hit_topics:
                    s += w.interest * min(1.0, hit_topics / n_interests)

            if p_len:
                ptoks = post_tokens.get(pid) or set()
                if ptoks:
                    inter = len(persona_tokens & ptoks)
                    if inter:
                        denom = p_len if p_len < len(ptoks) else len(ptoks)
                        s += w.persona * (inter / denom if denom else 0.0)

            s += w.random * rnd.random()

            if pid in seen:
                s *= w.seen_penalty
            scored.append((pid, s))

        if not scored:
            out[uid] = []
            continue

        k = min(top_k, len(scored))
        top = heapq.nlargest(k, scored, key=lambda x: x[1])
        out[uid] = top

    return out


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------
# 冷启动填充
# --------------------------------------------------------------------------


def fill_cold_start(
    recs: Dict[int, List[Tuple[int, float]]],
    user_ids: Iterable[int],
    all_post_ids: Sequence[int],
    top_k: int,
    rng: random.Random | None = None,
) -> Dict[int, List[Tuple[int, float]]]:
    """给推荐为空（或不足 top_k）的用户补上随机帖，保证每个 agent 都有内容可看。

    必须**补齐不足**，而不只是处理"完全为空"：否则早期轮次里部分智能体的
    feed 可能只有一两条，认知链缺少输入，个体会因此产生与自身画像无关的差异，
    污染后续的对比分析。
    """
    rnd = rng or random.Random()
    pool = list(all_post_ids)
    for uid in user_ids:
        cur = recs.setdefault(int(uid), [])
        if len(cur) >= top_k or not pool:
            continue
        have = {pid for pid, _ in cur}
        candidates = [p for p in pool if p not in have]
        if not candidates:
            continue
        need = min(top_k - len(cur), len(candidates))
        for pid in rnd.sample(candidates, need):
            cur.append((pid, 0.0))
    return recs
