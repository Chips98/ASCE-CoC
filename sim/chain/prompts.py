"""认知链 prompt 模板与输出解析。

两次调用契约
------------
第 1 次：**[情绪 / 情感 / 思维 / 立场]**，只输出思考，**不给任何工具**。
第 2 次：把第 1 次的完整思考原文放进 system，再输出
          **[观点 / 意图 / 一致性]** 并调用动作工具。

为什么这样切
------------
如果一次调用里同时给出"情绪推理"和"动作工具"，模型会倾向于**先决定动作、
再倒推一套情绪和立场来圆场**。这样得到的理由是动作的合理化解释，
无法用来回答"它为什么这么做"。拆成两次后，第 1 次物理上看不到工具，
只能做认知判断；第 2 次拿到的是已经写定、无法回改的认知链结论。
因果方向因此由**架构**保证，而不是靠提示模型"要诚实"。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .spec import (
    COGNITIVE_DIMENSIONS,
    DIMENSIONS,
    INTEGRATION_DIMENSION,
    STAGE1_KEYS,
    STAGE2_KEYS,
    ChainDimension,
    ChainNode,
    Profile,
    active_nodes,
    get_profile,
    node_index,
    total_node_count,
)

logger = logging.getLogger("asce.chain")

# 模型输出的标记符。选择"### 维度名"而不是纯"维度名："是为了让解析足够鲁棒，
# 同时保持人眼可读——模型偶尔漏掉方括号也会退化成"维度名："，解析器一并兼容。
DIM_PREFIX = "###"
ACTION_MARKER = "### 行动"


# ==========================================================================
# 输入上下文
# ==========================================================================


@dataclass
class ChainContext:
    """一次认知链推理所需的全部输入。

    注意 `interactions`（收到的定向互动）与 `feed_text`（推荐流）分开传：
    定向互动是外部注入的内容，若把它混进推荐流，"我看到的"和"有人对我说的"
    就分不开了，事后无法区分归因。这里显式分开，落库时也分开记录。
    """

    agent_id: int
    tick: int
    persona_text: str  # 用户画像（已渲染成文本）
    feed_text: str  # 本轮推荐流（帖子 + 评论树）渲染文本
    interactions: List[Dict[str, Any]] = field(default_factory=list)  # 定向互动
    memory_text: str = ""  # 历史记忆摘要
    recent_actions_text: str = ""  # 最近几轮自己做过什么
    topic_background: str = ""  # 话题背景
    platform_brief: str = ""  # 平台现状简报（热门话题等）


def _render_interactions(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "（本轮没有收到定向互动）"
    lines: List[str] = []
    for i, it in enumerate(items, 1):
        who = it.get("from_name") or it.get("from_persona") or f"用户{it.get('from_id', '?')}"
        content = (it.get("content") or "").strip()
        pid = it.get("post_id")
        where = f"（出现在帖子 {pid} 下）" if pid else ""
        lines.append(f"{i}. 来自「{who}」{where}：{content}")
    return "\n".join(lines)


# ==========================================================================
# 第 1 阶段：感受与自我
# ==========================================================================


def _node_instruction(dim: ChainDimension, node: ChainNode, stage: int) -> str:
    return (
        f"{node.key} {node.label}\n"
        f"    理论依据：{node.theory}\n"
        f"    你要想清楚：{node.observes}\n"
        f"    要写出来的内容：{node.outputs}"
    )


def _stage_block(profile: Profile, stage_keys: Tuple[str, ...]) -> str:
    """渲染某一阶段的维度与节点清单。"""
    chunks: List[str] = []
    for dim, nodes in active_nodes(profile, stage_keys):
        head = f"【{dim.label}】{dim.theory}\n  这一维在刻画什么：{dim.physics}"
        body = "\n".join(_node_instruction(dim, n, 0) for n in nodes)
        chunks.append(head + "\n" + body)
    return "\n\n".join(chunks)


def _output_contract(profile: Profile, stage_keys: Tuple[str, ...]) -> str:
    """生成给模型看的输出格式契约。"""
    lines: List[str] = []
    for dim, nodes in active_nodes(profile, stage_keys):
        lines.append(f"{DIM_PREFIX} {dim.label}")
        for n in nodes:
            lines.append(f"{n.label}：<这里写你的推理，1 到 3 句>")
        lines.append("")
    return "\n".join(lines).rstrip()


SYSTEM_STAGE1 = """你是一名社交媒体平台上的真实用户。你正在参与一场关于「{topic}」的公共讨论。

【最重要的一条】你的内部推理预算是有限的。**不要长篇自我检查、不要反复权衡措辞、
不要列出"我需要考虑哪些方面"**。想清楚就直接写出来。如果你把预算花在内部独白上，
结构化输出会被截断，这次交互就作废了。

在采取任何行动之前，你有一套固定的思考顺序，共四个维度。请严格沿着这个顺序思考，你的每一个维度都要真的受到上一个维度结论的影响。

【重要】这次你只需要思考，不需要决定做什么。平台动作会在下一步单独询问你。

{chain_block}

【必须遵守的要求】
1. 全文使用中文。不要输出英文句子，不要用英文写分析、检查或备注。
2. **不要写任何开场白、自我检查、字数统计或元评论**（例如"让我先看看""检查一下约束""字数合适""Final check"）。直接从第一个维度开始写，写满四个维度就立刻停笔。
3. 每个节点都要写出**针对这次看到的具体内容**的推理，1 到 3 句，不要写放之四海皆准的空话。
4. 各个维度之间要真的有推理推进关系。情绪要影响思维，思维要影响立场，而不是四段互不相干的独立描述。
5. 如果信息不足以形成某个判断，就直说"信息不足，暂时无法判断"，不要编造。
6. 这是思考脚手架，不是填空题。可以质疑它、可以写出与理论相反的实际情况。

【输出格式】严格按下面的格式输出，维度标题与节点名称必须原样保留，不要加任何额外标题：

{contract}"""


USER_STAGE1 = """{persona_block}

【话题背景】
{topic_background}

{memory_block}{recent_block}{platform_block}
【你本轮刷到的内容】
{feed_text}

【你收到的定向互动】
{interactions}

请按你的思考顺序，完成【{stage_label}】这几个维度的推理。只输出格式要求的内容。"""


# 第一阶段内容放在 user 消息里而不是 system 里：
# 实测放在 system 中时，模型会把"你已经完成了第一阶段的思考"读成一句说明，
# 而看不到具体内容，于是抱怨"没有显示第一阶段结论"并从头重新推理。
USER_STAGE2 = """【你本轮刷到的内容】（与第一阶段相同，供你核对 ID）
{feed_text}

【你收到的定向互动】
{interactions}

【你在第一阶段的思考结论】这是你自己刚才写下的判断，请把它当作已经成立的前提：
{stage1_output}

请接着完成【{stage_label}】的推理，然后调用工具执行你的行动。"""


SYSTEM_STAGE2 = """你是一名社交媒体平台上的真实用户，正在参与关于「{topic}」的公共讨论。

你刚刚已经完成了第一阶段的思考（情绪、情感、思维、立场），完整内容在下面这条用户消息里。**那是你自己写下的判断，是已经成立的前提，不要重新推理它。**

现在请继续完成剩下的三个维度。你已经形成的情绪、情感、思维与立场必须继续约束你的判断，不要与上面的结论自相矛盾。

{chain_block}

【可用工具】
{tool_catalog}

【必须遵守的要求】
1. 全文使用中文。不要输出英文句子，不要用英文写分析、检查或备注。
2. **不要写任何开场白、自我检查、字数统计或元评论**。直接从第一个维度开始写，写完三个维度立刻调用工具。
3. 每个节点都要写针对性推理，1 到 2 句，不要空话。
4. **必须调用工具**。如果确实什么都不想做，就调用 do_nothing，但不要只回复文字而不调用工具。
5. 你可以在本轮调用多次工具（例如先看帖子详情再评论）。
6. 你的动作必须能从上面的认知链推导出来。如果认知链与你想做的动作不一致，在【一致性】里如实说明，并交代你最终服从了哪一边。

【输出格式】先写思考，维度标题与节点名称必须原样保留；然后调用工具：

{contract}"""


# ==========================================================================
# 组装
# ==========================================================================


def build_stage1(
    ctx: ChainContext,
    profile: Profile,
    topic: str = "社会热点",
) -> List[Dict[str, str]]:
    """构造第 1 次调用的 messages。"""
    stage_label = "、".join(d.label for d, _ in active_nodes(profile, STAGE1_KEYS))
    system = SYSTEM_STAGE1.format(
        topic=topic,
        chain_block=_stage_block(profile, STAGE1_KEYS),
        contract=_output_contract(profile, STAGE1_KEYS),
    )
    user = USER_STAGE1.format(
        persona_block=ctx.persona_text,
        topic_background=ctx.topic_background or "（无额外背景）",
        memory_block=_block("【你记得的事】", ctx.memory_text),
        recent_block=_block("【你最近做过什么】", ctx.recent_actions_text),
        platform_block=_block("【平台现状】", ctx.platform_brief),
        feed_text=ctx.feed_text or "（推荐流为空）",
        interactions=_render_interactions(ctx.interactions),
        stage_label=stage_label,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_stage2(
    ctx: ChainContext,
    profile: Profile,
    stage1_output: str,
    tool_schemas: List[Dict[str, Any]],
    tool_catalog: str,
    topic: str = "社会热点",
) -> List[Dict[str, Any]]:
    """构造第 2 次调用的 messages。

    第一阶段的原文放在 **user** 消息里（而不是 system）：
    实测放在 system 中时模型会把它当成一句背景说明而"看不到"具体结论，
    进而从头重新推理第一阶段，浪费预算甚至自相矛盾。
    """
    stage_label = "、".join(d.label for d, _ in active_nodes(profile, STAGE2_KEYS))
    system = SYSTEM_STAGE2.format(
        topic=topic,
        chain_block=_stage_block(profile, STAGE2_KEYS),
        tool_catalog=tool_catalog,
        contract=_output_contract(profile, STAGE2_KEYS),
    )
    user = USER_STAGE2.format(
        feed_text=ctx.feed_text or "（推荐流为空）",
        interactions=_render_interactions(ctx.interactions),
        stage1_output=(stage1_output or "").strip() or "（第一阶段没有产出内容）",
        stage_label=stage_label,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _block(title: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    return f"{title}\n{body}\n\n"


# ==========================================================================
# 解析
# ==========================================================================


@dataclass
class ChainStep:
    """解析出来的一个节点。"""

    key: str
    dim: str  # 维度 key
    dim_label: str
    label: str
    text: str
    present: bool  # 这个节点是否有内容
    # 内容来源：True = 模型把节点名写出来了；False = 该维度的内容是一段没有
    # 标节点名的散文，由维度级兜底填充而来。区分这两者是必要的：
    # 否则"节点写出率"会被兜底掩盖，看不出模型有没有遵守输出契约。
    inherited: bool = False


@dataclass
class ParsedChain:
    """一次认知链输出的解析结果。"""

    steps: List[ChainStep] = field(default_factory=list)
    raw_text: str = ""
    action_text: str = ""  # "### 行动" 之后的内容（若模型写了）
    missing: List[str] = field(default_factory=list)  # 缺失的节点 key

    @property
    def explicit_count(self) -> int:
        """模型显式写出节点名的数量。"""
        return sum(1 for s in self.steps if s.present and not s.inherited)

    @property
    def inherited_count(self) -> int:
        return sum(1 for s in self.steps if s.present and s.inherited)

    def to_json(self) -> Dict[str, Any]:
        return {
            "steps": [
                {
                    "key": s.key,
                    "dim": s.dim,
                    "dim_label": s.dim_label,
                    "label": s.label,
                    "text": s.text,
                    "present": s.present,
                    "inherited": s.inherited,
                }
                for s in self.steps
            ],
            "missing": self.missing,
            "explicit_count": self.explicit_count,
            "inherited_count": self.inherited_count,
            "total_count": len(self.steps),
        }

    def text_by_dim(self) -> Dict[str, str]:
        """维度 key -> 该维度下各节点文本拼接（供落库宽表使用）。"""
        out: Dict[str, List[str]] = {}
        for s in self.steps:
            if s.present:
                out.setdefault(s.dim, []).append(s.text)
        return {k: " ".join(v) for k, v in out.items()}

    def stage_text(self) -> str:
        """把本阶段所有节点重新拼回一段可读文本（喂给第 2 阶段用）。"""
        lines: List[str] = []
        cur: Optional[str] = None
        for s in self.steps:
            if s.dim != cur:
                lines.append(f"{DIM_PREFIX} {s.dim_label}")
                cur = s.dim
            if s.present:
                lines.append(f"{s.label}：{s.text}")
        return "\n".join(lines)


def _split_action(text: str) -> Tuple[str, str]:
    """切出 "### 行动" 之后的部分。"""
    idx = text.find(ACTION_MARKER)
    if idx < 0:
        return text, ""
    return text[:idx], text[idx + len(ACTION_MARKER):].strip()


_NODE_LINE = re.compile(r"^\s*([^：:\n]{2,12})\s*[：:]\s*(.*)$")


def parse_chain(
    text: str,
    profile: Profile,
    stage_keys: Tuple[str, ...],
) -> ParsedChain:
    """从模型输出里解析出各节点文本。

    解析策略（由严到宽，尽量不丢信息）：
      1. 按 "### 标题" 切块。标题用**前缀匹配**：模型常写成
         "### 观点 - 态度强度"，用严格全等会把整段判为未命中；
      2. 块内按节点中文标签提取，兼容四种真实写法——
         "节点名：内容"、"### 维度 - 节点名：内容"、"**节点名**" 单独成行、
         以及多个节点合并在同一段（只用句号分隔）；
      3. 维度级兜底：若某维度的内容写成了一段没有标节点名的散文，
         该维度下未命中的节点由整段文本兜底填充，并标记 `inherited`；
      4. **不做跨维度全文兜底**：两个阶段存在同名节点，跨维度搜索会串味。

    未命中的节点记入 missing，用于在实时页上暴露"模型没按契约输出"。
    系统不代模型补默认值——否则分析时无法区分"模型真的这么想"与
    "系统替它填的"，节点写出率这个健康度指标也就失去意义了。
    """
    body, action_text = _split_action(text or "")
    idx = node_index(profile, stage_keys)
    # 节点标签 -> key，标签较长者优先，避免"情绪"误匹配"即时情绪"
    label_to_key = sorted(
        ((n.label, k) for k, n in idx.items()),
        key=lambda kv: -len(kv[0]),
    )

    # 只允许匹配本阶段启用的维度，避免第一阶段的维度名在第二阶段文本里
    # 被误当成分组标题（跨阶段同名节点会让解析结果串味）
    stage_dims = [d for d, _ in active_nodes(profile, stage_keys)]
    dim_by_label = {d.label: d for d in stage_dims}

    # 1) 按 "### 标题" 切块。标题用**前缀匹配**而不是全等：
    #    实测模型常写成 "### 观点 - 态度强度" 或 "### 意图：主观规范"，
    #    严格全等会把整段判为"未命中"。
    blocks: List[Tuple[Optional[str], str]] = []
    parts = re.split(r"^\s*#{2,4}\s*(.+?)\s*$", body, flags=re.M)
    if len(parts) == 1:
        blocks.append((None, body))
    else:
        if parts[0].strip():
            blocks.append((None, parts[0]))
        for i in range(1, len(parts), 2):
            blocks.append((parts[i].strip(), parts[i + 1] if i + 1 < len(parts) else ""))

    def _match_dim(heading: str) -> Optional[ChainDimension]:
        for label, dim in dim_by_label.items():
            if heading.startswith(label) or label in heading[:12]:
                return dim
        return None

    found: Dict[str, str] = {}
    inherited: Dict[str, str] = {}
    block_by_dim: Dict[str, str] = {}
    for heading, block_text in blocks:
        dim = _match_dim(heading) if heading else None
        if dim is None:
            continue
        block_by_dim.setdefault(dim.key, block_text)
        for node in dim.nodes:
            if node.key not in idx or node.key in found:
                continue
            val = _search_node(block_text, node.label)
            if val:
                found[node.key] = val

    # 2) 维度级兜底：模型把某维度的内容写成了一段没有标节点名的散文。
    #    实测第二阶段很常见（三个节点合成一段）。此时把整段内容作为该维度下
    #    **尚未命中**的节点的内容，并标记 inherited —— 既不丢信息，
    #    也能在实时页上把"显式标注"与"由维度段兜底填充"区分开。
    #    注意：不做跨维度全文兜底，否则第一阶段维度名会污染第二阶段。
    for dim, nodes in active_nodes(profile, stage_keys):
        block = block_by_dim.get(dim.key)
        if not block or not block.strip():
            continue
        for node in nodes:
            if node.key in found:
                continue
            inherited[node.key] = _clean(block)

    steps: List[ChainStep] = []
    missing: List[str] = []
    for dim, nodes in active_nodes(profile, stage_keys):
        for node in nodes:
            txt = found.get(node.key, "")
            present = bool(txt)
            was_inherited = False
            if not present and node.key in inherited:
                txt = inherited[node.key]
                present = True
                was_inherited = True
            if not present:
                missing.append(node.key)
            steps.append(
                ChainStep(
                    key=node.key,
                    dim=dim.key,
                    dim_label=dim.label,
                    label=node.label,
                    text=txt,
                    present=present,
                    inherited=was_inherited,
                )
            )
    return ParsedChain(
        steps=steps,
        raw_text=text or "",
        action_text=action_text,
        missing=missing,
    )


def _search_node(text: str, label: str) -> str:
    """在文本里找节点内容，返回清理后的内容（找不到则空串）。

    模型实际会混用四种写法，解析器必须全兼容（否则"节点写出率"这个健康指标
    会误报，进而掩盖真正的模型行为）：

    1. `即时情绪：内容` 独占一行
    2. `### 情绪 - 即时情绪`／`### 情绪-即时情绪：内容`（节点名嵌在标题里）
    3. `**即时情绪**` 单独成行，内容在下一行
    4. `态度：…。主观规范：…。感知行为控制：…` 多个节点合并成一段
       （实测第二阶段最常见：模型把三个节点写进一行，只用句号分隔）

    关键约束：节点名必须在**行首或紧跟句末标点**，否则正文里偶然提到
    "情感底色"这样的词会被误判成该节点已被写出。内容止于下一个节点名或换行。
    """
    if not text:
        return ""
    lead = rf"(?:^|\n|(?<=。)|(?<=；)|(?<=;)|(?<=！)|(?<=？))\s*(?:\*\*|#+\s*)?"

    # 形式 A：节点名是标题的末尾，内容在其后的行里（必须带 # 才算标题）
    pat0 = re.compile(
        rf"(?:^|\n)[ \t]*#+[^\n]*?{re.escape(label)}[ \t]*(?:\*\*)?[ \t]*\n"
        rf"(.+?)(?=\n[ \t]*#|\Z)",
        flags=re.S,
    )
    m0 = pat0.search(text)
    if m0 and m0.group(1).strip():
        return _clean(m0.group(1))

    # 形式 B：节点名 + 分隔符 + 内容（覆盖写法 1、2、4）
    pat = re.compile(
        rf"{lead}[\[【]?\s*{re.escape(label)}\s*[\]】]?\s*(?:\*\*)?\s*[：:\-—]\s*(.+?)"
        rf"(?=\s*(?:\*\*|#+\s*)?[\[【]?\s*[^\n：:\-—]{{2,10}}\s*[\]】]?\s*(?:\*\*)?\s*[：:\-—]"
        rf"|\n\s*#|\n\s*\n|\Z)",
        flags=re.S,
    )
    m = pat.search(text)
    if m and m.group(1).strip():
        return _clean(m.group(1))

    # 形式 C：标签单独成行（可加粗、可加方括号），内容在后续行
    pat2 = re.compile(
        rf"{lead}[\[【]?\s*{re.escape(label)}\s*[\]】]?\s*(?:\*\*)?\s*$\n"
        rf"(.+?)(?=\n\s*#|\n\s*\n|\Z)",
        flags=re.M,
    )
    m2 = pat2.search(text)
    if m2 and m2.group(1).strip():
        return _clean(m2.group(1))
    return ""


def _clean(value: str) -> str:
    """清掉模型常见的格式残留。"""
    v = value.strip()
    v = re.sub(r"^\*+|\*+$", "", v).strip()
    v = re.sub(r"^[-•]\s*", "", v).strip()
    v = re.sub(r"\s*\n\s*", " ", v)
    return v.strip()


def default_profile(key: str = "pro") -> Profile:
    return get_profile(key)


__all__ = [
    "ChainContext",
    "ChainStep",
    "ParsedChain",
    "build_stage1",
    "build_stage2",
    "parse_chain",
    "default_profile",
    "DIM_PREFIX",
    "ACTION_MARKER",
    "SYSTEM_STAGE1",
    "SYSTEM_STAGE2",
]
