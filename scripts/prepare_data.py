"""把 cut 数据集清洗成 ASCE-CoC 的输入格式。

背景：为什么必须清洗
--------------------
`participants.json` 里**认知标签泄漏在 6 个地方**，只删 `cognitive_profile`
是不够的：

1. `cognitive_profile`（四维枚举，8 个叶子字段）
2. `persona` —— 300/300 命中「立场|态度|观点|主张|倾向」，是标签的**自然语言
   渲染**。只删字段没用，必须整段重新生成。
3. `bio` —— 71/300 含态度词
4. `information_style` / `communication_style` / `media_trust` /
   `persuasion_susceptibility` —— 名字看着像"人设"，内容其实是**认知风格的
   标签**（如"数据驱动""情绪表达"），会直接提示模型该怎么思考
5. `realname` / `username` 里的 `伪装智能体（ID-N）` 是身份自指污染
6. `_synthetic` / `_enriched` 是数据工程内部标记

不清洗的后果：认知链还没开始推理，"我倾向于支持中方""我是分析型思维"就已经
写在人设里了，链上的每个节点都只是在复述输入——认知链会退化成一个昂贵的
模板填充器。

数据侧的两个坑
--------------
* **ID 命名空间碰撞**：real 用户 ID 1–499 ⊂ random 用户 ID 1–2250，按数字 ID
  关联会把大量帖子错误归属。清洗后统一发放新的 `user_id`（连续整数）。
  清洗过程内部会用上游编号做作者映射，但**写出前会剥离全部溯源字段**
  （见 `strip_source_fields`），公开数据不保留任何可回溯到上游账号的标识。
* **帖子池很小**：`posts.json` 2683 条里只有 167 条是顶层原创帖，其余是
  上下文帖与回复帖。若只保留原创帖，300 agent × 100 tick 需要 1800 倍复用。
  因此本脚本把**全部 2683 条**作为独立候选帖池，保留 `conversation_id` /
  `source_field` / `in_reply_to_id` 以便回放会话结构，并在报告里打印实际复用倍数。

用法：
    python scripts/prepare_data.py \
        --src /path/to/data_new/cut \
        --dst data/cut
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# 认知标签语汇黑名单：出现这些词的句子一律不进 persona / bio
# --------------------------------------------------------------------------
BAN_PATTERNS = [
    # 立场 / 态度
    r"立场", r"态度", r"观点", r"主张", r"倾向", r"支持中方", r"支持美方",
    r"亲美", r"反美", r"强硬", r"温和", r"中立", r"不明确", r"观望",
    # 情绪
    r"情绪", r"愤怒", r"恐惧", r"厌恶", r"喜悦", r"信任", r"惊讶", r"悲伤",
    r"期待", r"情感", r"易感", r"敏感",
    # 思维风格
    r"认知", r"思维", r"分析型", r"直觉型", r"数据驱动", r"理性分析",
    r"情绪驱动", r"情绪表达", r"数据论证", r"引用权威", r"权威依赖",
    r"说服", r"从众", r"批判",
    # 身份自指
    r"伪装智能体", r"智能体", r"该用户", r"此用户", r"本用户", r"ID-\d+", r"匿名用户",
]
BAN_RE = re.compile("|".join(BAN_PATTERNS))

# bio 用的**较窄**黑名单：只剔除真正会泄漏立场或思维基调的词。
# bio 是作者的自我介绍，像"关注中美贸易摩擦""热衷政治与经济话题"是客观陈述，
# 不该被删；但"倾向于支持中方""情绪化""分析型思维"必须删。
BIO_BAN_PATTERNS = [
    r"支持中方", r"支持美方", r"亲美", r"反美", r"倾向于支持", r"持.{0,4}立场",
    r"立场", r"态度", r"主张", r"不明确", r"中立", r"强硬", r"温和",
    r"情绪化", r"易受", r"易感", r"说服", r"从众",
    r"分析型", r"直觉型", r"数据驱动", r"情绪驱动", r"情绪表达",
    r"理性分析", r"数据论证", r"引用权威", r"权威依赖",
    r"愤怒", r"恐惧", r"厌恶", r"喜悦", r"惊讶", r"悲伤",
    r"伪装智能体", r"智能体", r"ID-\d+", r"匿名用户",
]
BIO_BAN_RE = re.compile("|".join(BIO_BAN_PATTERNS))

# 保留的人设字段（全部为客观描述，不含认知判断）
KEEP_FIELDS = [
    "age", "gender", "mbti", "country", "profession",
    "interested_topics", "influence_metrics", "economic_status", "education",
]


def sanitize_text(text: str, fallback: str = "", pattern: "re.Pattern[str] | None" = None) -> str:
    """按句剔除含认知标签的句子，保留其余描述。

    为什么要保留大部分内容：作者的自我介绍里有"关注中美贸易摩擦""热衷政治与
    经济话题"这类**纯客观的兴趣陈述**，它们对认知链是有用输入。全删会让所有
    人设退化成同一句模板，agent 之间失去差异。所以只剔除真正泄漏立场/情绪/
    思维风格的句子，并且：只有当剔除比例超过一半（说明原文基本是标签渲染）
    时才退回 fallback。
    """
    if not text:
        return fallback
    pat = pattern or BAN_RE
    parts = [s.strip() for s in re.split(r"[。；;！!？?\n]", str(text)) if s.strip()]
    if not parts:
        return fallback
    kept = [s for s in parts if not pat.search(s)]
    if not kept or len(kept) * 2 < len(parts):
        return fallback
    out = "。".join(kept)
    if not out.endswith("。"):
        out += "。"
    if len(out) < 12:
        return fallback
    return out


def build_persona(rec: Dict[str, Any], seq: int) -> str:
    """用客观字段重建角色速写。

    刻意写得"平"：只有人口学、职业、兴趣、影响力。凡是能直接提示立场、
    情绪基调或思维风格的内容都不进 persona——那是认知链要**推出来**的东西，
    不能预先把答案写进题面。
    """
    parts: List[str] = []
    country = rec.get("country") or "未知地区"
    age = rec.get("age") or "未知年龄段"
    gender = rec.get("gender") or ""
    prof = rec.get("profession") or ""
    edu = rec.get("education") or ""
    econ = rec.get("economic_status") or ""
    mbti = rec.get("mbti") or ""
    topics = rec.get("interested_topics") or []
    if isinstance(topics, str):
        topics = [t for t in re.split(r"[，,、]", topics) if t.strip()]

    seg = f"一位来自{country}的{age}"
    if gender and gender not in ("不愿透露", "非二元"):
        seg += f"的{gender}性"
    parts.append(seg + "。")
    if prof:
        parts.append(f"从事{prof}相关工作。")
    if edu:
        parts.append(f"受教育程度为{edu}。")
    if econ:
        parts.append(f"经济状况属于{econ}。")
    if mbti:
        parts.append(f"性格类型为 {mbti}。")
    if topics:
        parts.append("平时比较关注" + "、".join(str(t) for t in topics) + "这类话题。")

    infl = (rec.get("influence_metrics") or {}).get("influence_score")
    if infl:
        try:
            v = float(infl)
            if v >= 60:
                parts.append("在平台上发言比较有影响力，经常被转发。")
            elif v >= 20:
                parts.append("在平台上有一些固定的关注者。")
            else:
                parts.append("在平台上是一个普通的参与者。")
        except (TypeError, ValueError):
            pass

    parts.append("习惯用自己的方式表达看法。")
    return "".join(parts)


# 职业 -> 自然的自我描述片段，用于重建 bio（避免把源数据里带态度的描述搬进来）
PROFESSION_BIO: Dict[str, str] = {
    "政治": "平时会留意政策文件与官方表态",
    "科技": "工作中会接触到技术动态与产业新闻",
    "经济": "习惯跟踪市场数据与财经报道",
    "教育": "日常和学生、家长打交道比较多",
    "医疗": "工作节奏紧，关注面比较杂",
    "工程": "做事偏实务，喜欢看具体案例",
    "法律": "对规则和程序的细节比较在意",
    "媒体": "每天要过一遍各大平台的热点",
    "艺术": "对文化类话题比较有兴趣",
    "学生": "课余时间大多花在社交媒体上",
    "农业": "关心产地和价格的实际变化",
    "军事": "常看军事与安全方面的报道",
    "金融": "每天都会看盘和读研报",
    "销售": "接触的客户背景比较杂",
    "自由职业": "时间安排比较自由，上网时间多",
}

# 兴趣 -> 自然的措辞
TOPIC_BIO: Dict[str, str] = {
    "国际关系": "国际关系",
    "经济": "经济形势",
    "政治": "时政",
    "时政": "时政",
    "科技": "科技",
    "军事": "军事安全",
    "社会": "社会热点",
    "教育": "教育",
    "环境": "环境议题",
    "医疗": "医疗",
    "文化": "文化",
    "体育": "体育",
    "娱乐": "娱乐",
    "民生": "民生",
}


def build_bio(rec: Dict[str, Any], seq: int) -> str:
    """重建 bio：只写客观的兴趣与平台使用习惯。

    源 bio 里混着大量态度定调（"情绪敏感""倾向表达对国家利益的支持"
    "对中美贸易政策持批判性思考"），这些是认知链应当**推出来**的结论，
    写进题面就等于泄题。而"关注国际关系与科技发展"这类兴趣陈述是有用输入，
    必须保留，否则 300 个 agent 会退化成同一句模板、失去个体差异。
    """
    prof = str(rec.get("profession") or "").strip()
    topics = rec.get("interested_topics") or []
    if isinstance(topics, str):
        topics = [t for t in re.split(r"[，,、]", topics) if t.strip()]
    topic_txt = [TOPIC_BIO.get(str(t), str(t)) for t in topics][:3]

    parts: List[str] = []
    if topic_txt:
        # 用顿号连接，三种开头轮换，避免 300 条 bio 全是同一种句式
        joined = "、".join(topic_txt)
        head = ["关注", "比较关心", "平时常看"][seq % 3]
        parts.append(f"{head}{joined}方面的消息")
    else:
        parts.append("平时会浏览各类社会话题")

    if prof:
        seg = PROFESSION_BIO.get(prof)
        if seg:
            parts.append(seg)

    # 用影响力决定"平台角色"的措辞，这是可观测量而非态度
    infl = (rec.get("influence_metrics") or {}).get("influence_score")
    try:
        v = float(infl) if infl is not None else 0.0
    except (TypeError, ValueError):
        v = 0.0
    role_choices = (
        ["在平台上发言常被转载", "帖子经常被人引用"]
        if v >= 60
        else ["在平台上有一批固定的关注者", "发帖不算多但有人跟"]
        if v >= 20
        else ["在平台上主要是个看客，偶尔发言", "多数时候只是刷刷，不太发言"]
    )
    parts.append(role_choices[seq % 2])

    out = "，".join(parts) + "。"
    return out if len(out) >= 12 else "普通的社交媒体用户，关注公共议题。"


def prepare_profiles(src: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """清洗参与者池，并重新发放连续 user_id。"""
    raw: List[Dict[str, Any]] = src["participants"]
    out: List[Dict[str, Any]] = []
    dropped: Counter = Counter()

    for i, rec in enumerate(raw, 1):
        uid = i  # 重新发放的连续 ID，避开原始 ID 命名空间碰撞
        prof: Dict[str, Any] = {"user_id": uid, "display_name": f"用户{uid:04d}"}
        for k in KEEP_FIELDS:
            if k in rec:
                prof[k] = rec[k]

        # is_real：注意 `_synthetic` 的语义是"存在即为合成"，缺键才是真人
        prof["is_real"] = not bool(rec.get("_synthetic", False))

        prof["source_name"] = rec.get("username") or rec.get("realname") or ""
        m = re.search(r"ID-?(\d+)", str(prof["source_name"]))
        prof["source_id"] = int(m.group(1)) if m else None

        prof["bio"] = build_bio(rec, i)
        prof["persona"] = build_persona(rec, i)

        # 记录被丢弃的字段，便于审计
        for k in ("cognitive_profile", "information_style", "communication_style",
                  "media_trust", "persuasion_susceptibility", "_synthetic", "_enriched",
                  "follower_list", "follow_list", "realname", "username"):
            if k in rec:
                dropped[k] += 1

        out.append(prof)

    report = {
        "profiles_total": len(out),
        "real_count": sum(1 for p in out if p["is_real"]),
        "synthetic_count": sum(1 for p in out if not p["is_real"]),
        "dropped_fields": dict(dropped),
    }
    return out, report


def prepare_posts(src: Dict[str, Any], profiles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """清洗帖子池。

    把原始作者的 `source_id` 映射到新发放的 user_id；无法映射的作者
    （即不在参与者池里的真人）保留其 source_id 作为"外部作者"，并用
    负数 id 区分，避免与正式参与者冲突。
    """
    by_source: Dict[int, int] = {}
    for p in profiles:
        sid = p.get("source_id")
        if sid is not None:
            by_source.setdefault(int(sid), int(p["user_id"]))

    posts: List[Dict[str, Any]] = []
    unmapped: Counter = Counter()
    next_external = -1
    external_map: Dict[int, int] = {}

    for row in src["posts"]:
        raw_name = str(row.get("user_name") or "")
        m = re.search(r"ID-?(\d+)", raw_name)
        sid = int(m.group(1)) if m else None

        if sid is not None and sid in by_source:
            author_id = by_source[sid]
            author_external = False
        else:
            # 作者不在参与者池里：给它一个负数 id 占位，保证帖子正文可用
            if sid is None:
                sid = 0
            if sid not in external_map:
                external_map[sid] = next_external
                next_external -= 1
            author_id = external_map[sid]
            author_external = True
            unmapped[sid] += 1

        posts.append({
            "post_id": int(row["post_id"]),
            "author_id": author_id,
            "author_source_id": sid,
            "author_external": author_external,
            "content": (row.get("content") or "").strip(),
            "conversation_id": row.get("conversation_id"),
            "source_field": row.get("source_field"),
            "in_reply_to_id": row.get("in_reply_to_id"),
            "topic": row.get("topic"),
            "timestamp": row.get("timestamp"),
        })

    report = {
        "posts_total": len(posts),
        "by_source_field": dict(Counter(p.get("source_field") for p in posts)),
        "external_authors": len(external_map),
        "posts_by_external_author": int(sum(unmapped.values())),
        "empty_content": sum(1 for p in posts if not p["content"]),
        "min_len": min((len(p["content"]) for p in posts), default=0),
        "max_len": max((len(p["content"]) for p in posts), default=0),
    }
    return posts, report


#: 公开数据中不应保留的溯源字段。
#: `source_name` / `source_id` / `author_source_id` 指向上游数据集的原始用户标识，
#: 公开分发时属于可再识别线索（上游 ID 可能对应真实账号）。清洗阶段仍需它们来做
#: 作者映射，但**写出前必须剥离**。
SOURCE_FIELDS_PROFILE = ("source_name", "source_id")
SOURCE_FIELDS_POST = ("author_source_id",)


def strip_source_fields(
    profiles: List[Dict[str, Any]], posts: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """剥离溯源字段，返回新的对象列表（不改动入参）。"""
    out_p = [{k: v for k, v in r.items() if k not in SOURCE_FIELDS_PROFILE} for r in profiles]
    out_q = [{k: v for k, v in r.items() if k not in SOURCE_FIELDS_POST} for r in posts]
    return out_p, out_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始 cut 数据目录")
    ap.add_argument("--dst", required=True, help="输出目录")
    args = ap.parse_args()

    src_dir = os.path.abspath(args.src)
    dst_dir = os.path.abspath(args.dst)
    os.makedirs(dst_dir, exist_ok=True)

    participants = json.load(open(os.path.join(src_dir, "participants.json"), encoding="utf-8"))
    posts_raw = json.load(open(os.path.join(src_dir, "posts.json"), encoding="utf-8"))

    profiles, r1 = prepare_profiles({"participants": participants})
    posts, r2 = prepare_posts({"posts": posts_raw}, profiles)

    # 写盘前剥离上游溯源字段（作者映射已经完成，不再需要）
    profiles, posts = strip_source_fields(profiles, posts)
    r1["stripped_fields"] = list(SOURCE_FIELDS_PROFILE)
    r2["stripped_fields"] = list(SOURCE_FIELDS_POST)

    # ---- 校验：清洗后不应再出现认知标签 ----
    leaks: List[str] = []
    for p in profiles:
        for field in ("persona", "bio"):
            if BAN_RE.search(p.get(field) or ""):
                leaks.append(f"{p['user_id']}.{field}: {p[field][:60]}")

    # ---- 话题背景 ----
    bg_path = os.path.join(src_dir, "话题背景.txt")
    if os.path.exists(bg_path):
        background = open(bg_path, encoding="utf-8").read().strip()
    else:
        background = "中美关税战持续，双方在核心利益上互不相让，企业与消费者承压。"

    out_profiles = os.path.join(dst_dir, "profiles.json")
    out_posts = os.path.join(dst_dir, "posts.json")
    out_meta = os.path.join(dst_dir, "meta.json")

    json.dump(profiles, open(out_profiles, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(posts, open(out_posts, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(
        {"topic": "cut", "background": background, "profiles": r1, "posts": r2,
         "ban_leaks": leaks},
        open(out_meta, "w", encoding="utf-8"), ensure_ascii=False, indent=2,
    )

    print("=" * 70)
    print(f"数据清洗完成 → {dst_dir}")
    print("=" * 70)
    print(f"画像：{r1['profiles_total']} 条（真实 {r1['real_count']} / 合成 {r1['synthetic_count']}）")
    print(f"被丢弃的标签字段：{r1['dropped_fields']}")
    print(f"帖子：{r2['posts_total']} 条，按来源 {r2['by_source_field']}")
    print(f"      正文长度 {r2['min_len']}–{r2['max_len']} 字符，空正文 {r2['empty_content']} 条")
    print(f"      外部作者（不在参与者池）：{r2['external_authors']} 个，"
          f"其帖子 {r2['posts_by_external_author']} 条")
    print(f"标签泄漏残留：{len(leaks)} 处")
    for l in leaks[:10]:
        print("   !", l)
    print()
    print(f"产出：{out_profiles}\n      {out_posts}\n      {out_meta}")
    if leaks:
        sys.exit(2)


if __name__ == "__main__":
    main()
