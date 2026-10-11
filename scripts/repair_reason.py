#!/usr/bin/env python
"""按当前策略重算 `agent_turn.reason`。

背景
----
`reason` 是一个**派生索引字段**：它从认知链的某个节点摘取首句，方便在活动流里
一眼看到"为什么这么做"。完整推理始终存在 `agent_chain_step` 与
`stage1_text` / `stage2_text` 里，因此策略调整后可以完全离线重算，
不需要重跑仿真（重跑要花掉等量的推理算力）。

什么时候需要跑它
----------------
* 调整了 `SocialAgent._derive_reason` 的取值优先级（例如从"意图末节点"改为
  "意图首节点"）之后，希望历史库与新库口径一致；
* 早期版本写下的 `reason` 里混入了元推理式文本，想按新策略刷新。

用法：
    python scripts/repair_reason.py --db runs/xxx.db            # 先看会改多少
    python scripts/repair_reason.py --db runs/xxx.db --apply    # 实际写入
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.agent.agent import _first_sentence  # noqa: E402

#: 与 SocialAgent._derive_reason 保持一致的取值优先级。
#: 值为 True 表示取该维度的**第一个**节点（首节点更贴近"为什么想做"），
#: False 表示取**最后一个**节点（末节点是结论）。
PRIORITY: List[Tuple[str, bool]] = [
    ("intention", True),
    ("integration", False),
    ("opinion", False),
    ("stance", False),
]

#: agent_chain_step.dim 存的是维度 key（英文），与 spec.py 一致
DIM_LABEL = {
    "emotion": "情绪", "mood": "情感", "thinking": "思维",
    "stance": "立场", "opinion": "观点", "intention": "意图",
    "integration": "一致性",
}


def derive(steps_by_dim: Dict[str, List[str]]) -> str:
    for dim, first in PRIORITY:
        texts = steps_by_dim.get(dim) or []
        if not texts:
            continue
        return _first_sentence(texts[0] if first else texts[-1], 140)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--apply", action="store_true", help="实际写入（默认只做预览）")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """SELECT agent_id, tick, reason FROM agent_turn
           WHERE is_active=1 ORDER BY tick, agent_id"""
    )
    turns = cur.fetchall()

    cur.execute(
        """SELECT agent_id, tick, dim, text FROM agent_chain_step
           WHERE present=1 ORDER BY agent_id, tick, stage, step_order"""
    )
    chain: Dict[Tuple[int, int], Dict[str, List[str]]] = {}
    for r in cur.fetchall():
        key = (int(r["agent_id"]), int(r["tick"]))
        chain.setdefault(key, {}).setdefault(r["dim"] or "", []).append(r["text"] or "")

    changed = 0
    previews: List[Tuple[int, int, str, str]] = []
    updates: List[Tuple[str, int, int]] = []
    for t in turns:
        key = (int(t["agent_id"]), int(t["tick"]))
        new = derive(chain.get(key, {}))
        old = t["reason"] or ""
        if new and new != old:
            changed += 1
            updates.append((new, key[0], key[1]))
            if len(previews) < 6:
                previews.append((key[0], key[1], old, new))

    print(f"库：{args.db}")
    print(f"活跃轮次记录 {len(turns)} 条，其中 {changed} 条的 reason 会被更新\n")
    for aid, tick, old, new in previews:
        print(f"用户{aid} 第{tick}轮")
        print(f"  改前：{old[:80]}")
        print(f"  改后：{new[:80]}")
        print()

    if not args.apply:
        print("（预览模式，未写入。加 --apply 生效）")
        return

    cur.executemany(
        "UPDATE agent_turn SET reason=? WHERE agent_id=? AND tick=?", updates
    )
    conn.commit()
    print(f"已更新 {len(updates)} 条记录。")


if __name__ == "__main__":
    main()
