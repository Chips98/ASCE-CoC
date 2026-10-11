"""LLM 客户端：OpenAI 兼容接口，多端点轮询、并发限流、tool call、重试与统计。

关于工具调用的三个实测结论（vLLM + Qwen3 系列）
------------------------------------------------
* 服务端需开启 `--enable-auto-tool-choice --tool-call-parser qwen3_xml`；
  此后标准 OpenAI tools 协议可用，返回 `finish_reason="tool_calls"` 与结构化
  `tool_calls`。
* 模型发起 tool call 时 `content` 为 `None`，所以**思考必须与工具调用分离**，
  这正是本项目把认知链放在无工具的第一次调用里输出的原因。
* 多轮 tool loop（assistant.tool_calls → role=tool 结果 → 再决策）可用。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger("asce.llm")


class LLMError(RuntimeError):
    """LLM 调用失败的统一异常。"""


@dataclass
class EndpointStat:
    url: str
    calls: int = 0
    failures: int = 0
    total_latency: float = 0.0
    in_flight: int = 0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.calls if self.calls else 0.0


@dataclass
class LLMResponse:
    """一次调用的完整结果，思考与工具调用分开保存。"""

    content: str
    reasoning: str
    tool_calls: List[Dict[str, Any]]
    finish_reason: str
    latency: float
    endpoint: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    """异步 LLM 客户端。

    设计取舍
    --------
    * 用**每个端点一个信号量**做并发控制，而不是全局信号量：这样不同端点的
      吞吐互不干扰，也方便按端点排查是哪一张卡在拖后腿。
    * 端点轮询用**原子计数器 + 锁**，保证多协程并发下单次分配均匀，
      不会出现某个端点被连续派发而另一个闲置。
    * 连接类失败与业务类失败**分开计数**：前者（隧道抖动、服务端短暂停止
      accept）用更宽松的退避独自重试，后者（如 HTTP 400 参数错）立即抛出。
      混在一起会让一次网络抖动被误判为整轮失败。
    """

    def __init__(
        self,
        endpoints: List[str],
        model: str,
        api_key: str = "EMPTY",
        max_concurrent_per_endpoint: int = 8,
        timeout: float = 1800.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        connect_retries: int = 5,
        connect_backoff: float = 3.0,
    ) -> None:
        if not endpoints:
            raise ValueError("至少需要一个 LLM 端点")
        self.endpoints = [u.rstrip("/") for u in endpoints]
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        # 连接类失败（隧道抖动、服务端引擎短暂停顿）单独用更宽松的计数与退避：
        # 实测 vLLM 在高并发下会短暂停止 accept，1-2 秒后自愈；
        # 把它和"业务失败"混在一个重试计数里会导致整轮仿真误判为失败。
        self.connect_retries = int(connect_retries)
        self.connect_backoff = float(connect_backoff)

        self._session: Optional[aiohttp.ClientSession] = None
        self._sems = [asyncio.Semaphore(max_concurrent_per_endpoint) for _ in self.endpoints]
        self._rr = 0
        self._rr_lock = asyncio.Lock()
        self._stats: Dict[str, EndpointStat] = {u: EndpointStat(url=u) for u in self.endpoints}
        self.slow_calls = 0

    # ------------------------------------------------------------------ 生命周期
    async def __aenter__(self) -> "LLMClient":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def start(self) -> None:
        if self._session is None or self._session.closed:
            # 总超时由外层 asyncio.wait_for 控制，这里给一个宽松的连接超时
            timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=self.timeout)
            conn = aiohttp.TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=conn)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    # ------------------------------------------------------------------ 端点调度
    async def _next_endpoint(self) -> str:
        """轮询取下一个端点，返回其 URL。"""
        async with self._rr_lock:
            url = self.endpoints[self._rr % len(self.endpoints)]
            self._rr += 1
        return url

    def stats(self) -> Dict[str, Dict[str, Any]]:
        return {
            u: {
                "calls": s.calls,
                "failures": s.failures,
                "avg_latency": round(s.avg_latency, 2),
                "in_flight": s.in_flight,
            }
            for u, s in self._stats.items()
        }

    def in_flight(self) -> int:
        return sum(s.in_flight for s in self._stats.values())

    def health(self) -> Dict[str, Any]:
        """端点健康快照，供收尾报告与实时页展示。"""
        return {
            "endpoints": self.stats(),
            "slow_calls": self.slow_calls,
            "in_flight": self.in_flight(),
        }

    # ------------------------------------------------------------------ 主调用
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        enable_thinking: bool = True,
        tool_choice: str = "auto",
        response_format: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """发起一次对话补全，失败自动重试与换端点。"""
        if self._session is None:
            await self.start()

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format
        if extra:
            payload.update(extra)

        last_err: Optional[Exception] = None
        connect_fails = 0
        attempt = 0
        while attempt < self.max_retries:
            url = await self._next_endpoint()
            idx = self.endpoints.index(url)
            sem = self._sems[idx]
            stat = self._stats[url]
            async with sem:
                stat.in_flight += 1
                stat.calls += 1
                started = time.monotonic()
                try:
                    body = await self._post(url, payload)
                    latency = time.monotonic() - started
                    stat.total_latency += latency
                    if latency > 120:
                        self.slow_calls += 1
                        logger.warning(
                            "单次调用耗时 %.0fs（端点 %s，completion_tokens=%s）",
                            latency, url, (body.get("usage") or {}).get("completion_tokens"),
                        )
                    return self._parse(body, latency, url)
                except _ConnectError as exc:
                    # 连接类失败：不消耗常规重试预算，改用更宽松的退避
                    latency = time.monotonic() - started
                    stat.total_latency += latency
                    stat.failures += 1
                    last_err = exc
                    connect_fails += 1
                    logger.warning(
                        "LLM 端点不可达（第 %d 次，端点 %s）：%s；%.1fs 后重试",
                        connect_fails, url, exc, self.connect_backoff,
                    )
                    await asyncio.sleep(self.connect_backoff)
                    if connect_fails >= self.connect_retries:
                        attempt += 1
                        connect_fails = 0
                except _RetryableError as exc:
                    latency = time.monotonic() - started
                    stat.total_latency += latency
                    stat.failures += 1
                    last_err = exc
                    attempt += 1
                    logger.warning(
                        "LLM 可重试错误（第 %d/%d 次，端点 %s）：%s",
                        attempt, self.max_retries, url, exc,
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_backoff ** attempt)
                except Exception as exc:  # 业务级错误不重试
                    latency = time.monotonic() - started
                    stat.total_latency += latency
                    stat.failures += 1
                    raise LLMError(f"LLM 调用失败（端点 {url}）：{exc}") from exc
                finally:
                    stat.in_flight -= 1

        raise LLMError(f"LLM 调用失败（已重试 {self.max_retries} 次，连接类失败 {connect_fails} 次）：{last_err}")

    async def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert self._session is not None
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            async with self._session.post(
                f"{url}/chat/completions", json=payload, headers=headers
            ) as resp:
                text = await resp.text()
                if resp.status in (429, 500, 502, 503, 504):
                    raise _RetryableError(f"HTTP {resp.status}: {text[:300]}")
                if resp.status != 200:
                    raise LLMError(f"HTTP {resp.status}: {text[:500]}")
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    raise _RetryableError(f"响应不是合法 JSON：{text[:300]}") from exc
        except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError,
                aiohttp.ClientOSError) as exc:
            raise _ConnectError(f"连接失败：{exc}") from exc
        except aiohttp.ClientError as exc:
            raise _RetryableError(f"网络错误：{exc}") from exc
        except asyncio.TimeoutError as exc:
            raise _ConnectError("请求超时") from exc

    # ------------------------------------------------------------------ 响应解析
    @staticmethod
    def _parse(body: Dict[str, Any], latency: float, url: str) -> LLMResponse:
        choices = body.get("choices") or []
        if not choices:
            raise LLMError(f"响应没有 choices 字段：{json.dumps(body, ensure_ascii=False)[:300]}")
        choice = choices[0]
        msg = choice.get("message") or {}

        content = msg.get("content") or ""
        if isinstance(content, list):
            # 少数实现会把 content 返回成分段数组
            content = "".join(
                seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in content
            )

        # 思考内容：vLLM 可能放在 reasoning_content / reasoning，或者内联在 content 里。
        # 内联时要同时检查 `<think>` 与 `</think>`：实测 Qwen3 的 chat template
        # 会把开标签吞掉，只留 `</think>` 在 content 中——只判断 "<think" 会漏掉
        # 全部这种情况，导致内部推理被当成正文入库（实测污染了半数 turn）。
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        low = content.lower()
        if not reasoning and ("<think" in low or "</think" in low):
            content, reasoning = _split_think(content)

        tool_calls: List[Dict[str, Any]] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments")
            args: Any
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    args = {"__raw__": raw_args}
            else:
                args = raw_args or {}
            if not isinstance(args, dict):
                args = {"__raw__": args}
            tool_calls.append(
                {
                    "id": tc.get("id") or f"call_{len(tool_calls)}",
                    "name": fn.get("name") or "",
                    "arguments": args,
                    "arguments_raw": raw_args,
                }
            )

        usage = body.get("usage") or {}
        return LLMResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason") or "",
            latency=latency,
            endpoint=url,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            raw=body,
        )


class _RetryableError(RuntimeError):
    """可重试错误（限流、5xx、网络、超时、非法 JSON）。"""


class _ConnectError(_RetryableError):
    """连接类失败：隧道抖动、服务端引擎短暂停止 accept。

    单独成类是为了给它更宽松的退避策略——它与"模型返回了非法内容"这类
    可重试错误性质不同，用同一个计数会导致整轮仿真被误判为失败。
    """


def _split_think(text: str) -> tuple[str, str]:
    """把内联的 thinking 块从正文里剥出来，返回 (正文, 推理内容)。

    实测 Qwen3 在 `enable_thinking=true` 下会把内部推理**直接写进 content**，
    并以 `</think>` 收尾，之后才是真正的回答。如果不剥离，下游解析器会把整段
    内部推理（含 "Let me reconsider"、"No extra spaces or headers" 这类英文
    元评论）当成结构化输出来解析，严重污染认知链。

    两种情况必须区分：
    * 有 `<think>` 且有 `</think>` → 标签之后才是正文；
    * 有 `<think>` 但没有 `</think>` → 输出被长度截断，**正文一个字都没有**，
      此时返回空正文，让上层走"截断重试"而不是把推理当正文。
    """
    import re

    if not text:
        return text, ""
    low = text.lower()
    close = low.rfind("</think")
    if close >= 0:
        end = text.find(">", close)
        body = text[end + 1:] if end >= 0 else ""
        reasoning = text[:close]
        reasoning = re.sub(r"^\s*<think[^>]*>", "", reasoning, flags=re.I).strip()
        return body.strip(), reasoning.strip()
    open_m = re.search(r"<think[^>]*>", text, flags=re.I)
    if open_m:
        # 未闭合 = 被截断，没有可用正文
        return "", text[open_m.end():].strip()
    return text, ""


async def gather_limited(coros: List[Any], limit: int = 32) -> List[Any]:
    """带并发上限的 gather，返回结果列表；异常会以 Exception 形式返回。

    在端点信号量之外再加一层**任务级**限流：端点信号量只约束"同时在飞的请求数"，
    而一次性构造上千个协程对象本身也有内存与调度开销。
    这层限流保证协程是按需创建、逐步推进的。
    """
    sem = asyncio.Semaphore(limit)

    async def _run(coro: Any) -> Any:
        async with sem:
            try:
                return await coro
            except Exception as exc:  # 保持与 gather(return_exceptions=True) 一致
                return exc

    return await asyncio.gather(*[_run(c) for c in coros])
