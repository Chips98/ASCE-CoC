"""认知链两次调用的端到端验证脚本。

用途：在不启动完整仿真的前提下，单独验证
  1. 第 1 次调用能否稳定产出 4 个维度的推理链；
  2. 第 2 次调用能否读到第 1 次结论、产出剩余维度并**真的调用工具**；
  3. 解析器能否把节点完整提取出来（missing 应为空）。

用法：
    python scripts/test_chain.py --profile max
    python scripts/test_chain.py --profile flash --endpoint http://127.0.0.1:7862/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.agent.llm_client import LLMClient  # noqa: E402
from sim.chain.prompts import (  # noqa: E402
    ChainContext,
    build_stage1,
    build_stage2,
    parse_chain,
)
from sim.chain.spec import STAGE1_KEYS, STAGE2_KEYS, get_profile, total_node_count  # noqa: E402
from sim.tools.actions import openai_tool_schemas, tool_catalog_text  # noqa: E402

PERSONA = """【你的公开资料】
- 名称：伪装智能体（ID-108）
- 简介：关注国际关系与经济动态，主张理性对话，对中美贸易摩擦持复杂态度。
- 人口学：性别=女 | 年龄段=31-50 | MBTI=ISFP | 国家=中国 | 职业=政治
- 认知与社会特征：信息处理风格=数据驱动 | 沟通风格=情绪表达 | 经济阶层=高收入精英 | 受教育水平=高中及以下 | 媒体信任倾向=主流媒体 | 政治参与度=偶尔关注 | 易感性=高
- 关注话题：国际关系、经济
- Persona（角色速写）：她是一名 31-50 岁的中国女性政治工作者，性格内敛细腻，常通过数据驱动的方式分析中美关税战的影响。她倾向于情绪表达，尤其在社交媒体上对政策调整发出强烈呼吁。作为高收入精英，她关注国际关系与经济议题，但仅偶尔关注，且主要依赖主流媒体获取信息。尽管教育程度仅为高中及以下，她对政策的说服力却极为敏感。"""

FEED = """以下是平台推荐给你的内容（你可以点击帖子 ID 查看详情）。

【帖子 101】作者：用户 12（关注数 34）
中方在最新一轮关税谈判中坚持了底线，没有在核心利益上让步。
热度：赞 28 · 踩 3 · 评论 9

  └ 评论 501（作者 用户 27，赞 12）：强硬是对的，谈判本来就不该让步。
  └ 评论 502（作者 用户 41，赞 5）：但是企业端已经很吃力了，长期看成本会转嫁到消费者身上。

【帖子 102】作者：用户 8（关注数 120）
美国商务部宣布对部分中国商品加征关税，涉及金额约 180 亿美元。
热度：赞 41 · 踩 6 · 评论 15

  └ 评论 503（作者 用户 33，赞 20）：这会推高美国本土的通胀，他们自己也要付代价。
  └ 评论 504（作者 用户 9，赞 3）：数据上看，上一轮加征后中国对美出口下降了 12%。

【帖子 103】作者：用户 55（关注数 76）
有消息说双方可能在农产品采购上达成阶段性安排。
热度：赞 9 · 踩 1 · 评论 2

  └ 评论 505（作者 用户 61，赞 4）：又是传闻，等官方消息吧。"""

INTERACTIONS = [
    {
        "from_id": 77,
        "from_name": "温暖的朋友",
        "post_id": 101,
        "content": "我理解你对政策反复的疲惫，但从长期看，谈判桌上的坚持往往换来更稳的结果，别太焦虑。",
    }
]


async def run(profile_key: str, endpoint: str, model: str, temperature: float) -> int:
    profile = get_profile(profile_key)
    print(f"\n{'=' * 78}")
    print(f"认知链验证 | profile={profile.key}（{profile.label}）| 节点数={total_node_count(profile)}")
    print(f"端点={endpoint} | 模型={model}")
    print(f"{'=' * 78}\n")

    ctx = ChainContext(
        agent_id=108,
        tick=1,
        persona_text=PERSONA,
        feed_text=FEED,
        interactions=INTERACTIONS,
        topic_background="中美关税战持续，双方在核心利益上互不让步，企业端与消费者承压。",
        memory_text="上一轮我表达了对政策反复的担忧，有两个人回应了我。",
        recent_actions_text="第 0 轮：没有动作。",
    )

    client = LLMClient(
        endpoints=[endpoint],
        model=model,
        max_concurrent_per_endpoint=4,
        timeout=900,
        max_retries=2,
    )
    await client.start()

    exit_code = 0
    try:
        # ---------------- 第 1 次调用：不给工具 ----------------
        msgs1 = build_stage1(ctx, profile, topic="中美关税战")
        print(f"[第 1 次调用] 维度 = {[d.label for d, _ in __import__('sim.chain.spec', fromlist=['x']).active_nodes(profile, STAGE1_KEYS)]}")
        print(f"  system prompt 长度 = {len(msgs1[0]['content'])} 字")
        print(f"  user prompt 长度   = {len(msgs1[1]['content'])} 字")
        t0 = time.monotonic()
        resp1 = await client.chat(
            msgs1, tools=None, temperature=temperature, max_tokens=3000, enable_thinking=True
        )
        print(f"  耗时 {resp1.latency:.1f}s | finish={resp1.finish_reason} | "
              f"prompt_tokens={resp1.prompt_tokens} completion_tokens={resp1.completion_tokens}")
        print(f"  tool_calls（应为空）= {len(resp1.tool_calls)}")
        print("\n" + "-" * 78)
        print("第 1 阶段原始输出：")
        print("-" * 78)
        print(resp1.content.strip())
        print("-" * 78)

        parsed1 = parse_chain(resp1.content, profile, STAGE1_KEYS)
        print(f"\n[第 1 阶段解析] 命中 {len(parsed1.steps) - len(parsed1.missing)}/{len(parsed1.steps)} 节点"
              f" | missing={parsed1.missing}")
        if parsed1.missing:
            exit_code = 2

        # ---------------- 第 2 次调用：给工具 ----------------
        tools = openai_tool_schemas()
        msgs2 = build_stage2(
            ctx,
            profile,
            stage1_output=parsed1.stage_text(),
            tool_schemas=tools,
            tool_catalog=tool_catalog_text(),
            topic="中美关税战",
        )
        print(f"\n[第 2 次调用] 工具数={len(tools)} | system 长度={len(msgs2[0]['content'])} 字")
        resp2 = await client.chat(
            msgs2, tools=tools, temperature=temperature, max_tokens=3000, enable_thinking=True
        )
        print(f"  耗时 {resp2.latency:.1f}s | finish={resp2.finish_reason} | "
              f"prompt_tokens={resp2.prompt_tokens} completion_tokens={resp2.completion_tokens}")
        print("\n" + "-" * 78)
        print("第 2 阶段原始输出：")
        print("-" * 78)
        print(resp2.content.strip() or "（模型未输出正文，直接调用了工具）")
        print("-" * 78)
        print(f"\n工具调用 {len(resp2.tool_calls)} 次：")
        for tc in resp2.tool_calls:
            print(f"  → {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")

        parsed2 = parse_chain(resp2.content, profile, STAGE2_KEYS)
        print(f"\n[第 2 阶段解析] 命中 {len(parsed2.steps) - len(parsed2.missing)}/{len(parsed2.steps)} 节点"
              f" | missing={parsed2.missing}")
        if parsed2.missing:
            exit_code = 2
        if not resp2.tool_calls:
            print("\n[错误] 第 2 阶段没有调用任何工具")
            exit_code = 3

        # ---------------- 汇总 ----------------
        print(f"\n{'=' * 78}")
        print(f"总耗时 = {resp1.latency + resp2.latency:.1f}s")
        print(f"端点统计 = {client.stats()}")
        print(f"{'=' * 78}\n")
    finally:
        await client.close()

    return exit_code


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="max", choices=["max", "pro", "flash"])
    ap.add_argument("--endpoint", default="http://127.0.0.1:7862/v1")
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()
    code = asyncio.run(run(args.profile, args.endpoint, args.model, args.temperature))
    sys.exit(code)


if __name__ == "__main__":
    main()
