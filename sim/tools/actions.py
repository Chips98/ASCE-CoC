"""动作工具层：把 18 个社交媒体动作定义成 OpenAI function-calling 工具。

为什么用标准 tool call
----------------------
如果让模型在一个 JSON 字段里输出动作名与参数、再由后端做关键词匹配分发，
会有两个问题：参数不合法时只能静默丢弃（否则整轮失败），以及一轮只能落一个动作。
改用标准 tool call 后：

* 参数用 JSON Schema 严格约束，模型看得见类型与必填项；
* 执行器做参数与上下文校验，非法调用返回**结构化错误**，
  模型可以在同一个 agentic loop 里读到错误并改正；
* 同一轮允许**多次调用**（上限见 configs 的 max_tool_calls_per_tick），
  因此"先查看详情、再发表评论"这类多步行为可以自然发生。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# 动作分类：读动作不改变平台状态，写动作会改变平台状态
# --------------------------------------------------------------------------

READ_ACTIONS = frozenset(
    {
        "refresh",
        "trend",
        "search_posts",
        "search_user",
        "get_trending_comments",
        "get_post_detail",
        "get_user_profile",
    }
)

WRITE_ACTIONS = frozenset(
    {
        "like_post",
        "dislike_post",
        "create_comment",
        "reply_to_comment",
        "share_comment",
        "like_comment",
        "dislike_comment",
        "create_post",
        "follow",
        "unfollow",
        "do_nothing",
    }
)

ALL_ACTIONS = READ_ACTIONS | WRITE_ACTIONS


@dataclass(frozen=True)
class ToolSpec:
    """一个动作工具的完整定义。"""

    name: str
    cn_name: str  # 中文名，用于前端展示
    description: str  # 给模型看的说明（中文）
    parameters: Dict[str, Any]  # JSON Schema
    kind: str  # "read" | "write"
    max_calls_per_tick: int = 1  # 同一轮内该工具最多被调用几次


def _schema(props: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------
# 工具定义（给模型看的描述全部用中文，且说明"什么时候用"）
# --------------------------------------------------------------------------

TOOLS: List[ToolSpec] = [
    # ---------------- 写动作：与内容互动 ----------------
    ToolSpec(
        name="like_post",
        cn_name="点赞帖子",
        description="给一篇你认同或觉得有价值的帖子点赞。用于表达赞同、欣赏。",
        parameters=_schema(
            {"post_id": {"type": "integer", "description": "要点赞的帖子 ID"}},
            ["post_id"],
        ),
        kind="write",
        max_calls_per_tick=3,
    ),
    ToolSpec(
        name="dislike_post",
        cn_name="点踩帖子",
        description="给一篇你不认同或觉得没有价值的帖子点踩。用于表达反对。",
        parameters=_schema(
            {"post_id": {"type": "integer", "description": "要点踩的帖子 ID"}},
            ["post_id"],
        ),
        kind="write",
        max_calls_per_tick=3,
    ),
    ToolSpec(
        name="create_comment",
        cn_name="发表评论",
        description=(
            "在一篇帖子下发表你自己的评论。这是表达观点最主要的方式。"
            "content 必须是你在社交媒体上真的会对别人说的那句话："
            "直接谈关税、政策或事件本身，第一人称，口语化，1 到 3 句。"
            "不要写分析过程，不要声明自己持什么立场，不要复述思考结论，"
            "也不要出现认知链、维度、圈层期待我这类元描述。"
        ),
        parameters=_schema(
            {
                "post_id": {"type": "integer", "description": "要评论的帖子 ID"},
                "content": {
                    "type": "string",
                    "description": (
                        "评论正文。示例：这轮加税最后还是美国消费者买单，"
                        "我们这边的出口企业早就开始转口了。"
                    ),
                },
            },
            ["post_id", "content"],
        ),
        kind="write",
        max_calls_per_tick=2,
    ),
    ToolSpec(
        name="reply_to_comment",
        cn_name="回复评论",
        description=(
            "回复某一条已有的评论，用于与他人直接对话。"
            "content 直接说给那个人听，不要写分析或立场声明。"
        ),
        parameters=_schema(
            {
                "comment_id": {"type": "integer", "description": "要回复的评论 ID"},
                "content": {"type": "string", "description": "回复正文，中文，1 到 3 句口语"},
            },
            ["comment_id", "content"],
        ),
        kind="write",
        max_calls_per_tick=2,
    ),
    ToolSpec(
        name="share_comment",
        cn_name="转发评论",
        description="把一条你认为有价值的评论转发给你的关注者。",
        parameters=_schema(
            {"comment_id": {"type": "integer", "description": "要转发的评论 ID"}},
            ["comment_id"],
        ),
        kind="write",
        max_calls_per_tick=1,
    ),
    ToolSpec(
        name="like_comment",
        cn_name="点赞评论",
        description="给一条你认同的评论点赞。",
        parameters=_schema(
            {"comment_id": {"type": "integer", "description": "要点赞的评论 ID"}},
            ["comment_id"],
        ),
        kind="write",
        max_calls_per_tick=3,
    ),
    ToolSpec(
        name="dislike_comment",
        cn_name="点踩评论",
        description="给一条你不认同的评论点踩。",
        parameters=_schema(
            {"comment_id": {"type": "integer", "description": "要点踩的评论 ID"}},
            ["comment_id"],
        ),
        kind="write",
        max_calls_per_tick=3,
    ),
    ToolSpec(
        name="create_post",
        cn_name="发布新帖",
        description="发起一个新的讨论主题。只有当你确实有独立、值得单独成篇的看法时才使用，不要滥用。",
        parameters=_schema(
            {"content": {"type": "string", "description": "帖子正文，中文，2 到 5 句话"}},
            ["content"],
        ),
        kind="write",
        max_calls_per_tick=1,
    ),
    ToolSpec(
        name="follow",
        cn_name="关注用户",
        description="关注一个你希望持续看到其内容的用户。",
        parameters=_schema(
            {"followee_id": {"type": "integer", "description": "要关注的用户 ID"}},
            ["followee_id"],
        ),
        kind="write",
        max_calls_per_tick=2,
    ),
    ToolSpec(
        name="unfollow",
        cn_name="取消关注",
        description="取消关注一个你已经不想再看到其内容的用户。",
        parameters=_schema(
            {"followee_id": {"type": "integer", "description": "要取消关注的用户 ID"}},
            ["followee_id"],
        ),
        kind="write",
        max_calls_per_tick=1,
    ),
    ToolSpec(
        name="do_nothing",
        cn_name="不做任何动作",
        description="本轮只看不互动。当内容与你无关、或你不想表态时使用。如果确实什么都不想做，应当调用它，而不是不回复。",
        parameters=_schema({}, []),
        kind="write",
        max_calls_per_tick=1,
    ),
    # ---------------- 读动作：获取更多信息 ----------------
    ToolSpec(
        name="refresh",
        cn_name="刷新信息流",
        description="刷新推荐流，获取一批新的推荐帖子。",
        parameters=_schema({}, []),
        kind="read",
        max_calls_per_tick=1,
    ),
    ToolSpec(
        name="trend",
        cn_name="查看热门话题",
        description="查看当前平台上最热门、被讨论最多的话题。",
        parameters=_schema({}, []),
        kind="read",
        max_calls_per_tick=1,
    ),
    ToolSpec(
        name="search_posts",
        cn_name="搜索帖子",
        description="按关键词搜索相关帖子，用于主动寻找你关心的内容。",
        parameters=_schema(
            {"query": {"type": "string", "description": "搜索关键词，中文"}},
            ["query"],
        ),
        kind="read",
        max_calls_per_tick=1,
    ),
    ToolSpec(
        name="search_user",
        cn_name="搜索用户",
        description="按关键词搜索用户，用于找到值得关注的人。",
        parameters=_schema(
            {"query": {"type": "string", "description": "搜索关键词，中文"}},
            ["query"],
        ),
        kind="read",
        max_calls_per_tick=1,
    ),
    ToolSpec(
        name="get_trending_comments",
        cn_name="查看热门评论",
        description="查看当前平台上获得最多互动的评论。",
        parameters=_schema(
            {"limit": {"type": "integer", "description": "返回条数，默认 5"}},
            [],
        ),
        kind="read",
        max_calls_per_tick=1,
    ),
    ToolSpec(
        name="get_post_detail",
        cn_name="查看帖子详情",
        description="查看某一篇帖子的完整正文与它的评论树。当推荐流里的摘要不足以判断时使用。",
        parameters=_schema(
            {"post_id": {"type": "integer", "description": "要查看的帖子 ID"}},
            ["post_id"],
        ),
        kind="read",
        max_calls_per_tick=3,
    ),
    ToolSpec(
        name="get_user_profile",
        cn_name="查看用户资料",
        description="查看某个用户的公开资料，了解他的背景与关注话题。",
        parameters=_schema(
            {"user_id": {"type": "integer", "description": "要查看的用户 ID"}},
            ["user_id"],
        ),
        kind="read",
        max_calls_per_tick=3,
    ),
]

TOOL_BY_NAME: Dict[str, ToolSpec] = {t.name: t for t in TOOLS}


def openai_tool_schemas(names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """导出 OpenAI / vLLM 兼容的 tools 数组。

    names 为 None 时导出全部工具；否则只导出指定工具（保持 TOOLS 的原始顺序，
    顺序稳定有助于模型行为可复现）。
    """
    allow = set(names) if names is not None else None
    out: List[Dict[str, Any]] = []
    for t in TOOLS:
        if allow is not None and t.name not in allow:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
        )
    return out


def tool_catalog_text(cn_only: bool = False) -> str:
    """把工具清单渲染成纯文本，供 prompt 里做"可用动作"说明。

    保留这个纯文本目录是有意为之：模型即使不调用工具，也能在思考里正确指称
    动作，前端展示也用它做中文名映射。
    """
    lines: List[str] = []
    for t in TOOLS:
        if cn_only:
            lines.append(f"- {t.cn_name}")
            continue
        params = t.parameters.get("properties", {}) or {}
        if params:
            arg_desc = "、".join(
                f"{k}（{v.get('description', '')}）" for k, v in params.items()
            )
        else:
            arg_desc = "无需参数"
        tag = "只读" if t.kind == "read" else "会改变平台状态"
        lines.append(f"- {t.name}（{t.cn_name}，{tag}）：{t.description} 参数：{arg_desc}")
    return "\n".join(lines)


def validate_arguments(name: str, args: Dict[str, Any]) -> List[str]:
    """校验工具参数，返回错误信息列表（空列表表示通过）。

    只做必要的类型与必填校验，不做业务校验（业务校验在执行器里做，因为需要
    查库）。这样模型能拿到明确的错误文本并自我修正。
    """
    errors: List[str] = []
    spec = TOOL_BY_NAME.get(name)
    if spec is None:
        return [f"未知的工具名：{name}"]
    if not isinstance(args, dict):
        return [f"工具 {name} 的参数必须是一个 JSON 对象，收到的是 {type(args).__name__}"]

    props: Dict[str, Any] = spec.parameters.get("properties", {}) or {}
    required: List[str] = spec.parameters.get("required", []) or []

    for key in required:
        if key not in args or args[key] in (None, ""):
            errors.append(f"缺少必填参数：{key}")

    for key, value in args.items():
        if key not in props:
            errors.append(f"不认识的参数：{key}（本工具只接受 {sorted(props)}）")
            continue
        expected = props[key].get("type")
        if value is None:
            continue
        if expected == "integer" and not isinstance(value, int):
            # 允许模型传 "12" 这种字符串数字，但要在执行前统一转换
            if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                continue
            errors.append(f"参数 {key} 应为整数，收到 {type(value).__name__}")
        elif expected == "string" and not isinstance(value, str):
            errors.append(f"参数 {key} 应为字符串，收到 {type(value).__name__}")
        elif expected == "string" and isinstance(value, str) and not value.strip():
            errors.append(f"参数 {key} 不能是空白字符串")
    return errors


def coerce_arguments(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """把字符串数字转成整数，避免模型传 "12" 时下游炸掉。"""
    spec = TOOL_BY_NAME.get(name)
    if spec is None:
        return dict(args)
    props: Dict[str, Any] = spec.parameters.get("properties", {}) or {}
    out: Dict[str, Any] = {}
    for key, value in args.items():
        expected = (props.get(key) or {}).get("type")
        if expected == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            out[key] = int(value.strip())
        else:
            out[key] = value
    return out
