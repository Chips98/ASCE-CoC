"""认知链规格：6 个认知维度 + 一致性收口，三档 profile（max / pro / flash）。

设计要点
--------
1. 节点只描述「要做什么推理」，不规定「输出什么枚举值」。把认知约束成有限
   枚举会让推理退化为填空，还会附带一整套值域校验、中英混值与默认值兜底
   机制。本系统不让 agent 显式声明认知状态，认知状态只体现在自然语言
   思考里。
2. 链是**固定的**，不做逐 agent 的路径检索。个体差异由「人设 + 调节器」
   填充到各节点，而不是靠选不同路径。这样每轮 LLM 调用次数是常数级的。
3. 维度顺序即推理顺序，是物理保证的因果方向：
   情绪 → 情感 → 思维 → 立场 → 观点 → 意图 → 一致性
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# 数据模型
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainNode:
    """认知链上的一个推理节点。"""

    key: str  # 英文标识，如 "E1"，用于落库与聚合
    label: str  # 中文短标签，用于 prompt 标题
    theory: str  # 该节点依托的理论（自然语言）
    observes: str  # 该节点观察什么（给模型的指令）
    outputs: str  # 该节点应产出什么内容（自然语言，非枚举）


@dataclass(frozen=True)
class ChainDimension:
    """认知链上的一个维度。"""

    key: str  # 英文标识，如 "emotion"
    label: str  # 中文名，如 "情绪"
    theory: str  # 该维度的主理论
    physics: str  # 物理含义（一句话）
    icon: str  # 前端展示用的 emoji
    color: str  # 前端展示用的主色
    nodes: Tuple[ChainNode, ...]

    @property
    def node_keys(self) -> Tuple[str, ...]:
        return tuple(n.key for n in self.nodes)


@dataclass(frozen=True)
class Profile:
    """一档认知链密度配置。三档都统一走两次 LLM 调用。"""

    key: str  # max / pro / flash
    label: str  # 中文名
    desc: str
    # 维度 key -> 该档实际启用的节点 key 列表（空表示该维度本档不启用）
    plan: Dict[str, Tuple[str, ...]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 六个维度（正常版包含全部节点）
# --------------------------------------------------------------------------

EMOTION = ChainDimension(
    key="emotion",
    label="情绪",
    theory="普拉切克情绪之轮 + 罗素情绪环形模型",
    physics="即时的、短暂的情绪反应，由当下刺激直接触发，秒级到分钟级",
    icon="⚡",
    color="#E8892B",
    nodes=(
        ChainNode(
            key="E1",
            label="刺激评估",
            theory="评价理论：情绪由对事件的评价而非事件本身决定",
            observes="这条内容对我当下的目标、利益或价值意味着什么",
            outputs="判断它是威胁、机会，还是与我无关的中性信息；并说明它是否与我有关",
        ),
        ChainNode(
            key="E2",
            label="即时情绪",
            theory="普拉切克情绪之轮：喜悦 / 信任 / 恐惧 / 惊讶 / 悲伤 / 厌恶 / 愤怒 / 期待八种基本情绪",
            observes="八种基本情绪里，哪些被这条内容触发了，各自强度如何（弱 / 中 / 强）",
            outputs="点出被触发的具体情绪（可以不止一种）及其强度，说明为什么是这几种而不是别的",
        ),
        ChainNode(
            key="E3",
            label="效价与唤醒",
            theory="罗素情绪环形模型：任何情绪都可定位在效价（正负）× 唤醒（高低）两个轴上",
            observes="我此刻的情绪落在环形的哪个位置，是正向还是负向，是紧绷还是平静",
            outputs="一句话给出效价方向、唤醒高低，并说明它对我接下来判断的推力方向",
        ),
    ),
)

MOOD = ChainDimension(
    key="mood",
    label="情感",
    theory="积极与消极情感模型（PANAS）+ 贝克认知疗法（CBT）",
    physics="长期的情感底色，跨轮次稳定，是情绪反应的低通滤波器",
    icon="🌊",
    color="#3A7CA5",
    nodes=(
        ChainNode(
            key="M1",
            label="情感底色",
            theory="PANAS：以正性情感与负性情感两条独立轴描述长期情感倾向",
            observes="抛开这一刻的具体刺激，我这个人长期的情感底色是什么样",
            outputs="用自然语言描述我的长期情感倾向（偏正性 / 偏负性、活跃或低落、开放或戒备），并说明它此刻在放大还是抑制我刚被触发的情绪",
        ),
        ChainNode(
            key="M2",
            label="认知重构",
            theory="贝克认知疗法：自动想法往往是失真的，需要用证据去检验并重构",
            observes="我脑中第一时间冒出的自动想法是什么，它经得起证据检验吗",
            outputs="先写出自动想法，再用证据检验它，最后给出重构后的想法（若自动想法站得住，就明确说明它通过了检验）",
        ),
    ),
)

THINKING = ChainDimension(
    key="thinking",
    label="思维",
    theory="双重加工理论：System 1 快速直觉，System 2 缓慢分析",
    physics="信息处理风格，决定我用多少认知资源去加工这条内容",
    icon="🧠",
    color="#7B5EA7",
    nodes=(
        ChainNode(
            key="T1",
            label="加工模式",
            theory="双重加工理论：直觉系统快而省力但易受偏见影响，分析系统慢而费力但更准确",
            observes="这件事值得我动用分析系统吗，还是凭直觉就够了",
            outputs="说明我主要在用 System 1 直觉还是 System 2 分析，给出选择理由，并点出这种模式可能带来的偏差或优势",
        ),
        ChainNode(
            key="T2",
            label="证据与权衡",
            theory="证据累积与权衡：判断质量取决于调用了哪些知识、如何加权冲突证据",
            observes="我能调用哪些知识、经历或常识；有没有和主流说法冲突的证据",
            outputs="列出我用到的证据或经验（各自大概多重），说明冲突证据如何被加权，以及我承认哪些地方我其实不确定",
        ),
    ),
)

STANCE = ChainDimension(
    key="stance",
    label="立场",
    theory="道德基础理论（MFT）+ 右翼威权主义（RWA）",
    physics="价值取向层面的定性定位，离散、慢变，是观点的吸引子",
    icon="⚖️",
    color="#2E8B57",
    nodes=(
        ChainNode(
            key="S1",
            label="道德基础",
            theory="道德基础理论：关怀 / 公平 / 忠诚 / 权威 / 圣洁 五条直觉基础驱动道德与政治判断",
            observes="这条内容触动了我的哪几条道德基础，哪一条被强化、哪一条被冒犯",
            outputs="点出被触动的道德基础及其方向，说明它如何把判断推向某一侧",
        ),
        ChainNode(
            key="S2",
            label="合法性判断",
            theory="右翼威权主义：对权威、传统与群体共识的顺从倾向存在个体差异",
            observes="这里的权威、传统做法或群体共识，应该被尊重还是被质疑",
            outputs="给出我对权威 / 传统 / 群体共识的态度（尊重还是挑战）及理由，并说明这种倾向在我身上是强还是弱",
        ),
        ChainNode(
            key="S3",
            label="立场定位",
            theory="态度形成：价值取向经判断收敛为可表述的立场",
            observes="综合上面两条，我在这件事上究竟站在哪一边",
            outputs="用一句自然语言说清我的立场，并可说明我的立场是否与过去一致、有没有松动",
        ),
    ),
)

OPINION = ChainDimension(
    key="opinion",
    label="观点",
    theory="社会判断理论（接受—拒绝纬度）+ 态度强度理论",
    physics="对眼前具体对象的认同或反对强度，连续、可变，对具体刺激敏感",
    icon="🎯",
    color="#C1453B",
    nodes=(
        ChainNode(
            key="O1",
            label="接受与拒绝纬度",
            theory="社会判断理论：每个人在议题上有一个接受纬度、一个拒绝纬度和一个不表态纬度",
            observes="周边有哪些可能的说法，哪些落在我能接受的范围内，哪些已经越过我的拒绝阈值",
            outputs="分别说出我能接受的说法、我拒绝的说法，以及我不表态的说法，并指出眼前这条内容落在哪一区",
        ),
        ChainNode(
            key="O2",
            label="态度强度",
            theory="态度强度理论：态度的确定性、重要性与可及性决定它能否抵抗说服",
            observes="我对这个立场的确定性有多高、它对我有多重要、我多容易想起来",
            outputs="给出确定性、重要性、可及性三个方面的强弱判断，并说明这会让我更固执还是更容易被说动",
        ),
        ChainNode(
            key="O3",
            label="观点变化方向",
            theory="说服与同化对比：落在接受纬度的信息被同化，落在拒绝纬度的信息引发对比效应",
            observes="看完这条内容，我的观点是更坚定了，还是出现了裂缝",
            outputs="明确说出我的观点是固化、松动还是被反转（也可以说没有变化），并指认是哪一点造成了这个变化",
        ),
    ),
)

INTENTION = ChainDimension(
    key="intention",
    label="意图",
    theory="计划行为理论（TPB）",
    physics="行动倾向，由态度、主观规范与感知行为控制三者共同决定",
    icon="🚀",
    color="#1F6FB2",
    nodes=(
        ChainNode(
            key="I1",
            label="态度",
            theory="计划行为理论之行为态度：对行为本身的正面或负面评价",
            observes="如果我真的去做（评论、点赞、转发或什么也不做），我内心怎么评价这个行为本身",
            outputs="说明我对将要采取的行为持正面还是负面评价，以及理由",
        ),
        ChainNode(
            key="I2",
            label="主观规范",
            theory="计划行为理论之主观规范：感知到的他人期待与群体压力",
            observes="我所在的圈子期待我怎么做，我会不会因为别人的看法而调整",
            outputs="说明我感知到的群体压力有多大、方向如何，以及它对我行为的影响程度",
        ),
        ChainNode(
            key="I3",
            label="感知行为控制",
            theory="计划行为理论之感知行为控制：自认为能否顺利执行该行为",
            observes="我有没有能力、把握和条件去把这个行为做出来",
            outputs="说明我自认为的掌控程度（高 / 中 / 低），以及有什么现实约束",
        ),
    ),
)

INTEGRATION = ChainDimension(
    key="integration",
    label="一致性",
    theory="认知一致性：多个认知维度之间需要收口，冲突必须被显式报告",
    physics="跨维度的自检，保证行为有可追溯的认知来源，同时暴露认知与行为的不一致",
    icon="🔗",
    color="#5A6B7B",
    nodes=(
        ChainNode(
            key="Z1",
            label="认知链一致性",
            theory="认知一致性理论：情绪、情感、思维、立场、观点、意图之间应可相互解释",
            observes="从即时情绪一路走到行为意图，这条链自洽吗？哪一步出现了张力",
            outputs="声明链条是否自洽；若不自洽，明确指出是哪两个维度冲突（例如立场支持但情绪厌烦），并说明本轮最终服从了哪一个",
        ),
    ),
)

#: 全部维度，顺序即推理顺序
DIMENSIONS: Tuple[ChainDimension, ...] = (
    EMOTION,
    MOOD,
    THINKING,
    STANCE,
    OPINION,
    INTENTION,
    INTEGRATION,
)

#: 六个认知维度（不含一致性收口）
COGNITIVE_DIMENSIONS: Tuple[ChainDimension, ...] = DIMENSIONS[:-1]

#: 一致性收口维度
INTEGRATION_DIMENSION: ChainDimension = INTEGRATION


# --------------------------------------------------------------------------
# 两个阶段的切分
# --------------------------------------------------------------------------
# 第一阶段「感受与自我」：只输出思维链文本，**不暴露任何动作工具**。
# 这样模型无法先决定动作再倒推情绪，因果方向由架构保证。
STAGE1_KEYS: Tuple[str, ...] = ("emotion", "mood", "thinking", "stance")

# 第二阶段「判断与行动」：读取第一阶段的完整结论，产出观点、意图、一致性，
# 并在此处才暴露动作工具。
STAGE2_KEYS: Tuple[str, ...] = ("opinion", "intention", "integration")


# --------------------------------------------------------------------------
# 三档 profile
# --------------------------------------------------------------------------

PROFILES: Dict[str, Profile] = {
    "max": Profile(
        key="max",
        label="完整版",
        desc="17 节点全开，用于论文主实验、案例深挖与机理分析",
        plan={
            "emotion": ("E1", "E2", "E3"),
            "mood": ("M1", "M2"),
            "thinking": ("T1", "T2"),
            "stance": ("S1", "S2", "S3"),
            "opinion": ("O1", "O2", "O3"),
            "intention": ("I1", "I2", "I3"),
            "integration": ("Z1",),
        },
    ),
    "pro": Profile(
        key="pro",
        label="精简版",
        desc="13 节点，用于大规模长轮次实验，保留每个维度的主干",
        plan={
            "emotion": ("E1", "E2"),
            "mood": ("M1", "M2"),
            "thinking": ("T1", "T2"),
            "stance": ("S1", "S3"),
            "opinion": ("O2", "O3"),
            "intention": ("I1", "I2"),
            "integration": ("Z1",),
        },
    ),
    "flash": Profile(
        key="flash",
        label="极简版",
        desc="7 节点，用于 smoke、回归测试与超大规模试跑",
        plan={
            "emotion": ("E2",),
            "mood": ("M1",),
            "thinking": ("T1",),
            "stance": ("S3",),
            "opinion": ("O2",),
            "intention": ("I2",),
            "integration": ("Z1",),
        },
    ),
}


def get_profile(key: str) -> Profile:
    """按 key 取 profile，非法 key 回退到 pro 并给出提示。"""
    if key in PROFILES:
        return PROFILES[key]
    raise KeyError(f"未知的认知链 profile: {key!r}，可选 {sorted(PROFILES)}")


def active_nodes(profile: Profile, stage_keys: Tuple[str, ...]) -> List[Tuple[ChainDimension, List[ChainNode]]]:
    """取出某阶段在给定 profile 下实际启用的 (维度, 节点列表)。"""
    out: List[Tuple[ChainDimension, List[ChainNode]]] = []
    for dim in DIMENSIONS:
        if dim.key not in stage_keys:
            continue
        enabled = profile.plan.get(dim.key, ())
        nodes = [n for n in dim.nodes if n.key in enabled]
        if nodes:
            out.append((dim, nodes))
    return out


def node_index(profile: Profile, stage_keys: Tuple[str, ...]) -> Dict[str, ChainNode]:
    """key -> ChainNode，用于解析模型输出时校验节点是否齐全。"""
    idx: Dict[str, ChainNode] = {}
    for dim, nodes in active_nodes(profile, stage_keys):
        for n in nodes:
            idx[n.key] = n
    return idx


def total_node_count(profile: Profile) -> int:
    return sum(len(v) for v in profile.plan.values())
