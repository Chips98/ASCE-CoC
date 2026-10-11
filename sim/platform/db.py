"""数据库连接与建表。

单一 schema 来源
----------------
全部表结构只由 `schema.sql` 一处定义，启动时一次建全。
把建表逻辑分散在多个模块各自 `CREATE TABLE` 会带来一个具体风险：
某张表在特定执行路径下缺失时，写入失败会连带事务回滚，
把同一事务内的主记录一起丢掉。集中一份 schema 可以消除这类问题。

外键与幂等
----------
这里显式开启 `PRAGMA foreign_keys`，但**不用外键约束做业务保证**：
仿真的写入顺序天然乱序，靠 PK / UNIQUE 约束保证幂等更可靠。
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger("asce.db")

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# 需要初始化的表清单，供自检使用
ALL_TABLES = [
    "user", "post", "comment",
    "post_like", "post_dislike", "comment_reaction", "follow", "comment_share",
    "rec", "trace",
    "agent_turn", "agent_chain_step", "agent_tool_call", "agent_interaction",
    "intervention", "campaign_target",
    "sim_meta", "sim_tick_stat",
]


def connect(db_path: str | Path, read_only: bool = False) -> sqlite3.Connection:
    """打开一个配置好的连接。"""
    path = str(db_path)
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30.0, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if not read_only:
        # WAL 让"仿真在写、实时页在读"能并发，这是实时监控页的前提
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA cache_size=-64000")
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.execute("PRAGMA foreign_keys=ON")
        # 仿真写入是单进程多协程，busy_timeout 防止实时页读锁冲突
        cur.execute("PRAGMA busy_timeout=30000")
    cur.close()
    return conn


def create_all(conn: sqlite3.Connection) -> None:
    """执行 schema.sql，幂等建表，并补齐后加的列。"""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)
    _migrate(conn)
    logger.info("数据库表已就绪（共 %d 张目标表）", len(ALL_TABLES))


# 后加的列：CREATE TABLE IF NOT EXISTS 不会给已存在的表补列，
# 因此这里显式做一次幂等迁移，让旧库也能继续用。
_LATE_COLUMNS = [
    ("agent_chain_step", "inherited", "INTEGER DEFAULT 0"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for table, column, decl in _LATE_COLUMNS:
        cur.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cur.fetchall()}
        if not existing:
            continue
        if column not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            logger.info("已为 %s 补齐列 %s", table, column)
    cur.close()


def table_counts(conn: sqlite3.Connection, tables: Optional[Iterable[str]] = None) -> Dict[str, int]:
    """统计各表行数，供收尾自检与实时页使用。"""
    out: Dict[str, int] = {}
    cur = conn.cursor()
    for name in (tables or ALL_TABLES):
        try:
            cur.execute(f"SELECT COUNT(*) FROM {name}")
            out[name] = int(cur.fetchone()[0])
        except sqlite3.Error:
            out[name] = -1
    cur.close()
    return out


def get_meta(conn: sqlite3.Connection, key: str, default: Optional[str] = None) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("SELECT value FROM sim_meta WHERE key=?", (key,))
    row = cur.fetchone()
    cur.close()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO sim_meta(key, value, updated_at) VALUES(?,?,datetime('now')) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, str(value)),
    )


def fetch_all(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    return rows


def fetch_one(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    cur.close()
    return row


def execute(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> int:
    cur = conn.cursor()
    cur.execute(sql, params)
    rowid = cur.lastrowid
    cur.close()
    return int(rowid or 0)


def executemany(conn: sqlite3.Connection, sql: str, seq: Sequence[Sequence[Any]]) -> None:
    if not seq:
        return
    cur = conn.cursor()
    cur.executemany(sql, seq)
    cur.close()


def backup(src_path: str | Path, dst_path: str | Path) -> None:
    """把仿真库快照到另一个文件，供实时页只读访问，避免读写互相干扰。"""
    src = sqlite3.connect(str(src_path))
    dst = sqlite3.connect(str(dst_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
