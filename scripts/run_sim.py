#!/usr/bin/env python
"""ASCE-CoC 仿真入口。

用法：
    python scripts/run_sim.py --config configs/smoke.yaml

实时监控页会在仿真启动时自动在配置的端口打开（默认 http://127.0.0.1:8801/）。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.simulation import run_simulation  # noqa: E402


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # 第三方库降噪
    for noisy in ("asyncio", "aiohttp", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser(description="ASCE-CoC 多智能体社会认知仿真")
    ap.add_argument("--config", required=True, help="配置文件路径")
    ap.add_argument("--verbose", action="store_true", help="输出调试日志")
    args = ap.parse_args()

    setup_logging(args.verbose)
    # 放到独立线程里跑 asyncio：某些环境（Jupyter / 已嵌入的循环）里
    # asyncio.run 会与既有事件循环冲突，用线程隔离最稳。
    import threading

    box: Dict[str, Any] = {}

    def _worker() -> None:
        try:
            box["code"] = asyncio.run(run_simulation(args.config))
        except KeyboardInterrupt:
            box["code"] = 130
        except BaseException as exc:  # noqa: BLE001
            box["exc"] = exc

    t = threading.Thread(target=_worker, name="asce-sim")
    t.start()
    try:
        while t.is_alive():
            t.join(0.5)
    except KeyboardInterrupt:
        print("\n收到中断信号，等待收尾…")
        t.join(30)
        box.setdefault("code", 130)
    if "exc" in box:
        raise box["exc"]
    sys.exit(int(box.get("code", 0)))


if __name__ == "__main__":
    main()
