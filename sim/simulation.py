"""仿真主循环。

一个 tick 的顺序
----------------
1. 推进仿真时钟
2. 发话题帖。作者按**兴趣匹配**指派给参与者，让每轮新增的话题由最可能关心它的
   人发出——这样推荐流里的"关注"信号才有实际含义
3. 重算推荐表（`rec`）
4. 按激活率抽活跃 agent
5. 并发执行每个活跃 agent 的完整因果链（两次 LLM 调用 + tool loop）
6. 未激活的 agent 也写一条 turn 记录，保留"活跃/非活跃"对比的基线
7. 落 tick 统计 → 实时页通过 DB 读到这里的数据

范围裁剪
--------
主循环只承载"内容生产 → 推荐 → 决策 → 落库 → 统计"这条主链路，因此：

* 干预（intervention）保留表结构，但默认不接入主循环，作为独立课题另行启用；
* 不做配额抽样，也不加载认知值域空间——参与者池由配置文件直接给定；
* 不写无上限的文本流水日志，运行统计统一落 `sim_tick_stat` 表；
* 不做 CSV 输出，全部结果只落 SQLite。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .agent.agent import SocialAgent
from .agent.llm_client import LLMClient, gather_limited
from .chain.spec import PROFILES, get_profile
from .platform import db as dbm
from .platform.clock import SimClock
from .platform.core import Platform
from .platform.recsys import RecsysWeights
from .persistence.store import TurnStore

logger = logging.getLogger("asce.sim")


@dataclass
class SimConfig:
    """已校验的运行配置。"""

    raw: Dict[str, Any]
    topic: str
    profiles_path: str
    posts_path: str
    background: str
    db_path: str

    num_agents: int
    num_timesteps: int
    activate_prob: float
    activate_prob_range: Optional[Tuple[float, float]]
    round_post_num: int
    seed_source_fields: List[str]
    pool_source_fields: List[str]
    max_rec_post_len: int
    refresh_post_count: int
    max_visible_comments: int
    allow_self_rating: bool
    bootstrap_follows: int
    clock_step_minutes: int

    chain_profile: str
    stage1_max_tokens: int
    stage2_max_tokens: int
    enable_thinking: bool
    max_tool_calls_per_tick: int
    max_tools_per_call: int
    memory_rounds: int

    endpoints: List[str]
    model: str
    api_key: str
    temperature: float
    max_concurrent_per_endpoint: int
    timeout: float
    max_retries: int

    recsys_weights: RecsysWeights

    live: Dict[str, Any]
    seed: int

    @staticmethod
    def load(path: str) -> "SimConfig":
        raw = yaml.safe_load(open(path, encoding="utf-8"))
        root = Path(__file__).resolve().parent.parent

        data = raw.get("data") or {}
        posts_path = _abspath(root, data["posts"])
        meta_path = _abspath(root, data.get("background_file") or "")
        background = ""
        if meta_path and os.path.exists(meta_path):
            try:
                background = json.load(open(meta_path, encoding="utf-8")).get("background") or ""
            except (OSError, ValueError):
                background = ""

        run = raw.get("run") or {}
        out_dir = _abspath(root, run.get("output_dir") or "runs")
        os.makedirs(out_dir, exist_ok=True)
        name = run.get("name") or "sim"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_path = os.path.join(out_dir, f"{name}_{ts}.db")

        sim = raw.get("simulation") or {}
        chain = raw.get("cognitive_chain") or {}
        inf = raw.get("inference") or {}
        rec = raw.get("recsys") or {}

        rng = sim.get("activate_prob_range")
        rng_t = (float(rng[0]), float(rng[1])) if isinstance(rng, (list, tuple)) and len(rng) == 2 else None

        endpoints = inf.get("endpoints") or ["http://127.0.0.1:7862/v1"]
        if isinstance(endpoints, str):
            endpoints = [x.strip() for x in endpoints.split(",") if x.strip()]

        profile_key = str(chain.get("profile") or "pro")
        if profile_key not in PROFILES:
            raise ValueError(f"未知的认知链档位 {profile_key!r}，可选 {sorted(PROFILES)}")

        return SimConfig(
            raw=raw,
            topic=str(data.get("topic") or "cut"),
            profiles_path=_abspath(root, data["profiles"]),
            posts_path=posts_path,
            background=background,
            db_path=db_path,
            num_agents=int(sim.get("num_agents") or 10),
            num_timesteps=int(sim.get("num_timesteps") or 5),
            activate_prob=float(sim.get("activate_prob") or 1.0),
            activate_prob_range=rng_t,
            round_post_num=int(sim.get("round_post_num") or 5),
            seed_source_fields=list(sim.get("seed_source_fields") or ["original_post"]),
            pool_source_fields=list(sim.get("pool_source_fields")
                                    or ["original_post", "context_post", "target_post"]),
            max_rec_post_len=int(sim.get("max_rec_post_len") or 10),
            refresh_post_count=int(sim.get("refresh_post_count") or 6),
            max_visible_comments=int(sim.get("max_visible_comments") or 3),
            allow_self_rating=bool(sim.get("allow_self_rating", False)),
            bootstrap_follows=int(sim.get("bootstrap_follows") or 3),
            clock_step_minutes=int(sim.get("clock_step_minutes") or 30),
            chain_profile=profile_key,
            stage1_max_tokens=int(chain.get("stage1_max_tokens") or 6000),
            stage2_max_tokens=int(chain.get("stage2_max_tokens") or 6000),
            enable_thinking=bool(chain.get("enable_thinking", True)),
            max_tool_calls_per_tick=int(chain.get("max_tool_calls_per_tick") or 3),
            max_tools_per_call=int(chain.get("max_tools_per_call") or 3),
            memory_rounds=int(chain.get("memory_rounds") or 3),
            endpoints=endpoints,
            model=str(inf.get("model") or "qwen3.5-9b"),
            api_key=str(inf.get("api_key") or "EMPTY"),
            temperature=float(inf.get("temperature") or 0.7),
            max_concurrent_per_endpoint=int(inf.get("max_concurrent_per_endpoint") or 8),
            timeout=float(inf.get("timeout") or 1800),
            max_retries=int(inf.get("max_retries") or 3),
            recsys_weights=RecsysWeights(**(rec.get("weights") or {})),
            live=raw.get("live") or {},
            seed=int(run.get("seed") or 42),
        )


def _abspath(root: Path, p: str) -> str:
    if not p:
        return ""
    q = Path(p)
    return str(q if q.is_absolute() else (root / q))


class Simulation:
    """一次仿真的全部状态。"""

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.clock = SimClock(step_minutes=cfg.clock_step_minutes)
        self.platform = Platform(
            db_path=cfg.db_path,
            clock=self.clock,
            topic=cfg.topic,
            recsys_weights=cfg.recsys_weights,
            max_rec_post_len=cfg.max_rec_post_len,
            refresh_post_count=cfg.refresh_post_count,
            max_visible_comments=cfg.max_visible_comments,
            allow_self_rating=cfg.allow_self_rating,
            seed=cfg.seed,
        )
        self.store = TurnStore(self.platform.conn)
        self.llm: Optional[LLMClient] = None
        self.agents: Dict[int, SocialAgent] = {}

        # 数据池
        self.profiles: List[Dict[str, Any]] = []
        self.chosen: List[Dict[str, Any]] = []   # 本次实际注册的参与者（连续 user_id）
        self.posts: List[Dict[str, Any]] = []
        self.seed_pool: List[Dict[str, Any]] = []
        self.feed_pool: List[Dict[str, Any]] = []
        self._seed_cursor = 0

        self.stats: Dict[str, Any] = {
            "ticks": [],
            "tool_calls": 0,
            "failed_turns": 0,
            "chain_missing": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    # ==================================================================
    # 准备
    # ==================================================================
    def load_data(self) -> None:
        cfg = self.cfg
        self.profiles = json.load(open(cfg.profiles_path, encoding="utf-8"))
        self.posts = json.load(open(cfg.posts_path, encoding="utf-8"))
        seed_fields = set(cfg.seed_source_fields)
        pool_fields = set(cfg.pool_source_fields)
        self.seed_pool = [p for p in self.posts if p.get("source_field") in seed_fields]
        self.feed_pool = [p for p in self.posts if p.get("source_field") in pool_fields]
        if not self.feed_pool:
            raise RuntimeError("推荐候选池为空，请检查 simulation.pool_source_fields")
        if not self.seed_pool:
            # 退回全池当种子，避免配置写错时直接崩
            logger.warning("种子帖池为空，退回使用全部候选帖作为种子")
            self.seed_pool = list(self.feed_pool)

        rng = self.rng
        rng.shuffle(self.profiles)
        chosen = self.profiles[: cfg.num_agents]

        dbm.set_meta(self.platform.conn, "topic", cfg.topic)
        dbm.set_meta(self.platform.conn, "num_agents", cfg.num_agents)
        dbm.set_meta(self.platform.conn, "num_timesteps", cfg.num_timesteps)
        dbm.set_meta(self.platform.conn, "chain_profile", cfg.chain_profile)
        dbm.set_meta(self.platform.conn, "seed", cfg.seed)
        dbm.set_meta(self.platform.conn, "model", cfg.model)
        dbm.set_meta(self.platform.conn, "background", cfg.background)
        dbm.set_meta(self.platform.conn, "pool_size", len(self.feed_pool))
        dbm.set_meta(self.platform.conn, "seed_pool_size", len(self.seed_pool))
        dbm.set_meta(self.platform.conn, "db_path", cfg.db_path)

        # 注册参与者（重新分配连续 user_id：1..N）
        self.chosen: List[Dict[str, Any]] = []
        for i, prof in enumerate(chosen, start=1):
            prof = dict(prof)
            prof["user_id"] = i
            prof.setdefault("display_name", f"用户{i:04d}")
            self.platform.register_user(i, prof, step=0)
            self.chosen.append(prof)

        # 外部作者占位用户：让帖子表里的 user_id 都能查到，避免显示 "用户-1"
        authors = sorted({int(p["author_id"]) for p in self.feed_pool if int(p["author_id"]) < 0})
        for aid in authors:
            self.platform.register_user(
                aid,
                {"username": f"外部用户{abs(aid)}", "display_name": f"外部用户{abs(aid)}",
                 "persona": "这条内容来自你并不认识的平台用户。", "bio": "平台上的其他用户。",
                 "is_control": True},
                step=0,
            )

        logger.info(
            "数据就绪：参与者 %d 人（真实 %d）、候选帖 %d 条、种子帖池 %d 条、外部作者 %d 个",
            cfg.num_agents,
            sum(1 for p in chosen if p.get("is_real")),
            len(self.feed_pool), len(self.seed_pool), len(authors),
        )

    def seed_initial_content(self) -> int:
        """把候选帖池导入 post 表（tick 0）。

        这样做的原因：语料是**历史发言**，仿真从一开始就应该处在一个已有讨论的
        环境里，而不是白板起步。若只在每轮新发 `round_post_num` 条，前几轮智能体
        几乎无内容可看，认知链缺少输入。
        导入时按 tick 0 的沙盒时间写入，时间衰减对它们一致。
        """
        rows: List[Tuple[Any, ...]] = []
        for p in self.feed_pool:
            rows.append((
                int(p["post_id"]),
                int(p["author_id"]),
                p["content"],
                self.cfg.topic,
                self.clock.stamp(),
                0,
                self.rng.randint(0, 3),   # num_likes：给一点初始热度差异
                0,
                p.get("conversation_id"),
                json.dumps(
                    {"source_field": p.get("source_field"),
                     "in_reply_to_id": p.get("in_reply_to_id"),
                     "source_post_id": p.get("post_id")},
                    ensure_ascii=False,
                ),
            ))
        dbm.executemany(
            self.platform.conn,
            """INSERT OR IGNORE INTO post(post_id,user_id,content,topic,created_at,step_number,
                                         num_likes,num_dislikes,conversation_id,extra_json)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        dbm.executemany(
            self.platform.conn,
            """INSERT OR IGNORE INTO post_like(user_id,post_id,created_at,step_number)
               VALUES(?,?,?,?)""",
            [
                (int(p["author_id"]), int(p["post_id"]), self.clock.stamp(), 0)
                for p in self.feed_pool if self.rng.random() < 0.3
            ],
        )
        dbm.set_meta(self.platform.conn, "seeded_posts", len(rows))
        logger.info("已导入 %d 条历史语料作为 tick 0 的初始内容", len(rows))
        return len(rows)

    def build_agents(self) -> None:
        profile = get_profile(self.cfg.chain_profile)
        for uid, prof in sorted(self.agents_profile_map().items()):
            self.agents[uid] = SocialAgent(
                agent_id=uid,
                profile=prof,
                platform=self.platform,
                llm=self.llm,
                store=self.store,
                chain_profile=profile,
                topic=self.cfg.topic,
                topic_background=self.cfg.background,
                max_tool_calls=self.cfg.max_tool_calls_per_tick,
                memory_rounds=self.cfg.memory_rounds,
                temperature=self.cfg.temperature,
                stage1_max_tokens=self.cfg.stage1_max_tokens,
                stage2_max_tokens=self.cfg.stage2_max_tokens,
                enable_thinking=self.cfg.enable_thinking,
                max_tools_per_call=self.cfg.max_tools_per_call,
            )

    def agents_profile_map(self) -> Dict[int, Dict[str, Any]]:
        """user_id -> 已注册画像（load_data 里已重新分配连续 user_id）。"""
        return {int(p["user_id"]): p for p in self.chosen}

    # ==================================================================
    # tick 内动作
    # ==================================================================
    def emit_seed_posts(self, tick: int) -> int:
        """发话题帖。作者按**兴趣匹配**指派给参与者。

        选作者的方式是：按帖子正文与该智能体关注话题的重叠度打分，
        在同轮候选中取最高者（带轻微随机扰动，避免总是同一人发帖）。
        帖子因此由"最可能关心它的人"发出，既符合直觉，
        也让推荐流的 follow 信号有意义。
        """
        from .platform.recsys import DEFAULT_TOPIC_KEYWORDS, interest_match

        n = min(self.cfg.round_post_num, max(1, len(self.seed_pool)))
        made = 0
        uid_list = [i for i in range(1, self.cfg.num_agents + 1)]
        for _ in range(n):
            if not self.seed_pool:
                break
            item = self.seed_pool[self._seed_cursor % len(self.seed_pool)]
            self._seed_cursor += 1
            content = item.get("content") or ""
            if not content.strip():
                continue

            best_uid, best_score = None, -1.0
            sample = self.rng.sample(uid_list, min(8, len(uid_list)))
            for uid in sample:
                prof = self.chosen[uid - 1]
                s = interest_match(content, prof.get("interested_topics") or [],
                                   DEFAULT_TOPIC_KEYWORDS)
                s += self.rng.random() * 0.05  # 轻微扰动，避免总是同一个人发
                if s > best_score:
                    best_uid, best_score = uid, s
            author = best_uid or self.rng.choice(uid_list)
            res = self.platform.create_post(
                author, content, is_seed=True,
                source_post_id=int(item["post_id"]),
                conversation_id=item.get("conversation_id"),
                extra={"source_field": item.get("source_field"),
                       "origin_post_id": item.get("post_id")},
                step=tick,
            )
            if res.get("success"):
                made += 1
        return made

    # ==================================================================
    # 主流程
    # ==================================================================
    async def run(self) -> int:
        cfg = self.cfg
        print("=" * 78)
        print(f"ASCE-CoC 仿真启动")
        print(f"  话题      : {cfg.topic}")
        print(f"  规模      : {cfg.num_agents} agents × {cfg.num_timesteps} ticks")
        print(f"  认知链档位: {cfg.chain_profile}（{PROFILES[cfg.chain_profile].label}，"
              f"{sum(len(v) for v in PROFILES[cfg.chain_profile].plan.values())} 节点）")
        print(f"  模型      : {cfg.model} @ {', '.join(cfg.endpoints)}")
        print(f"  数据库    : {cfg.db_path}")
        print("=" * 78)

        self.load_data()
        self.seed_initial_content()

        # 初始关注网络
        made = self.platform.bootstrap_follows(
            list(range(1, cfg.num_agents + 1)), k=cfg.bootstrap_follows, rng=self.rng, step=0
        )
        logger.info("已建立 %d 条初始关注关系（按兴趣相似度）", made)

        # 实时页（独立子进程，仿真结束后仍可复盘）
        live_srv = None
        if cfg.live.get("enabled"):
            try:
                # 用绝对导入：通过 scripts/run_sim.py 以脚本方式启动时，`sim`
                # 并未被当作包加载（__package__ 为空），此时
                # `from ..live.server import ...` 会抛
                # "attempted relative import beyond top-level package"。
                from sim.live.server import start_live_server
            except ImportError:
                from .live.server import start_live_server  # 包内导入时的回退
            try:
                live_srv = start_live_server(cfg.db_path, cfg.live, self.platform.conn)
                print(f"\n  实时监控页 → {live_srv.url}\n")
            except Exception as exc:
                logger.warning("实时页启动失败（不影响仿真）：%s", exc)

        self.llm = LLMClient(
            endpoints=cfg.endpoints,
            model=cfg.model,
            api_key=cfg.api_key,
            max_concurrent_per_endpoint=cfg.max_concurrent_per_endpoint,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
        )
        await self.llm.start()
        self.build_agents()

        # tick 0：只更新推荐表，不做决策
        self.platform.set_tick(0)
        self.platform.update_recs(list(self.agents.keys()), step=0)
        self.store.save_tick_stat(0, {"active_count": 0, "post_count": len(self.posts)})

        ok = True
        try:
            for tick in range(1, cfg.num_timesteps + 1):
                t0 = time.monotonic()
                self.platform.set_tick(tick)
                print(f"\n── tick {tick}/{cfg.num_timesteps} "
                      f"({self.clock.stamp()}) " + "─" * 40)

                seeded = self.emit_seed_posts(tick)
                self.platform.update_recs(list(self.agents.keys()), step=tick)

                # 激活抽样
                if cfg.activate_prob_range:
                    p = self.rng.uniform(*cfg.activate_prob_range)
                else:
                    p = cfg.activate_prob
                n_active = max(1, int(len(self.agents) * p))
                active_ids = set(self.rng.sample(list(self.agents.keys()),
                                                 min(n_active, len(self.agents))))

                # 并发跑认知链
                tasks = [self.agents[uid].run_tick(tick) for uid in sorted(active_ids)]
                # 并发执行：任务级上限 = 端点数 × 单端点并发，让所有端点都吃满
                results = await gather_limited(
                    tasks,
                    limit=max(1, len(cfg.endpoints) * cfg.max_concurrent_per_endpoint),
                )

                failed = sum(1 for r in results if hasattr(r, "status") and r.status == "failed")
                missing = 0
                tool_n = 0
                s1 = s2 = pt = ct = 0
                for r in results:
                    if not hasattr(r, "status"):
                        logger.error("agent 任务异常: %r", r)
                        failed += 1
                        continue
                    tool_n += len(r.tool_calls)
                    pt += r.prompt_tokens
                    ct += r.completion_tokens
                    s1 += r.stage1_ms
                    s2 += r.stage2_ms
                    for ch in (r.chain1, r.chain2):
                        if ch is not None:
                            missing += len(ch.missing)

                # 非活跃 agent 也留痕
                for uid in sorted(set(self.agents.keys()) - active_ids):
                    self.agents[uid].run_tick_skipped(tick)

                elapsed = time.monotonic() - t0
                stat = self.platform.tick_stat(tick)
                self.store.save_tick_stat(tick, {
                    "active_count": len(active_ids),
                    "post_count": stat["posts"],
                    "comment_count": stat["comments"],
                    "like_count": stat["likes"],
                    "follow_count": stat["follows"],
                    "tool_call_count": tool_n,
                    "chain_missing_count": missing,
                    "stage1_ms_total": s1,
                    "stage2_ms_total": s2,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "elapsed_sec": elapsed,
                })
                self.stats["ticks"].append({
                    "tick": tick, "active": len(active_ids), "seeded": seeded,
                    "failed": failed, "missing": missing, "tool_calls": tool_n,
                    "elapsed": elapsed, "tokens": pt + ct,
                })
                self.stats["tool_calls"] += tool_n
                self.stats["failed_turns"] += failed
                self.stats["chain_missing"] += missing
                self.stats["prompt_tokens"] += pt
                self.stats["completion_tokens"] += ct

                print(f"   发帖 {seeded}｜激活 {len(active_ids)}｜工具调用 {tool_n}｜"
                      f"缺节点 {missing}｜失败 {failed}｜耗时 {elapsed:.1f}s")
        except KeyboardInterrupt:
            print("\n收到中断，正在收尾…")
            ok = False
        finally:
            if self.llm is not None:
                print("\n==== LLM 端点统计 ====")
                for url, st in self.llm.stats().items():
                    print(f"  {url}  调用={st['calls']} 失败={st['failures']} "
                          f"平均延迟={st['avg_latency']}s")
                await self.llm.close()

        self.report()
        if live_srv is not None:
            print(f"\n实时监控页仍在运行：{live_srv.url}（按 Ctrl+C 结束）")
            try:
                while True:
                    await asyncio.sleep(3600)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            live_srv.stop()
        return 0 if ok else 1

    # ==================================================================
    # 收尾报告
    # ==================================================================
    def report(self) -> None:
        conn = self.platform.conn
        print("\n" + "=" * 78)
        print("仿真统计")
        print("=" * 78)
        counts = dbm.table_counts(conn, [
            "user", "post", "comment", "post_like", "follow", "rec", "trace",
            "agent_turn", "agent_chain_step", "agent_tool_call", "sim_tick_stat",
        ])
        for name, n in counts.items():
            print(f"  {name:<20} {n}")
        print()
        for t in self.stats["ticks"]:
            print(f"  tick {t['tick']:>3}｜激活 {t['active']:>3}｜发帖 {t['seeded']:>3}｜"
                  f"工具 {t['tool_calls']:>3}｜缺节点 {t['missing']:>3}｜{t['elapsed']:.1f}s")
        print(f"\n  总工具调用 {self.stats['tool_calls']}｜失败 turn {self.stats['failed_turns']}｜"
              f"缺失节点 {self.stats['chain_missing']}｜"
              f"token {self.stats['prompt_tokens']}+{self.stats['completion_tokens']}")

        # 重复曝光率：cut 语料只有 2683 条，这是最能反映仿真健康度的指标
        row = dbm.fetch_one(
            conn,
            "SELECT COUNT(*) AS n, COUNT(DISTINCT post_id) AS d FROM rec",
        )
        if row and row["n"]:
            reuse = row["n"] / max(row["d"], 1)
            print(f"  推荐曝光 {row['n']} 次，覆盖 {row['d']} 条帖，平均复用 {reuse:.1f} 倍")
        print("=" * 78)


async def run_simulation(config_path: str) -> int:
    cfg = SimConfig.load(config_path)
    sim = Simulation(cfg)
    return await sim.run()
