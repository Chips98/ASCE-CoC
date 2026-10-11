"""仿真时钟：把 tick 映射成沙盒时间戳。

**时钟是仿真时间的唯一权威**。系统里不保留第二套时间来源（例如另用环境变量
维护一个 tick 计数），否则两套时间各自自增、互不校准，依赖时间戳的特征
（如推荐的时间衰减）会静默失效。这里 tick 只由调用方显式推进，
`now()` 与 `at()` 都从同一处换算。
"""

from __future__ import annotations

from datetime import datetime, timedelta

# 沙盒纪元：与真实数据的时间戳风格保持一致（选择 2025-04-09 是为了让
# 从 cut 数据引入的 historical 时间戳看起来自然）
SANDBOX_START = datetime(2025, 4, 9, 10, 0, 0)


class SimClock:
    """一个 tick 对应沙盒里的 `step_minutes` 分钟。"""

    def __init__(
        self,
        start: datetime | None = None,
        step_minutes: int = 30,
    ) -> None:
        self.start = start or SANDBOX_START
        self.step_minutes = max(1, int(step_minutes))
        self.tick = 0

    def set_tick(self, tick: int) -> None:
        self.tick = int(tick)

    def now(self) -> datetime:
        return self.start + timedelta(minutes=self.step_minutes * self.tick)

    def at(self, tick: int) -> datetime:
        return self.start + timedelta(minutes=self.step_minutes * int(tick))

    def stamp(self) -> str:
        return self.now().strftime("%Y-%m-%d %H:%M:%S")

    def elapsed_hours(self, since: datetime) -> float:
        return (self.now() - since).total_seconds() / 3600.0
