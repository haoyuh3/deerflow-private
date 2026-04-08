"""Rate-limiting middleware for the DeerFlow lead agent.

Implements a dual-bucket token-bucket algorithm:
  1. Global bucket  — shared across every thread on this process.
  2. Per-thread bucket — isolated per LangGraph thread_id.

Both buckets refill continuously at their configured RPM rate.
The middleware fires in ``before_model`` so the check happens *before* each
LLM call, not after a potentially expensive tool execution.

Usage (via config.yaml):
    rate_limit:
      enabled: true
      global_rpm: 60
      thread_rpm: 10
      retry_wait_seconds: 1.0   # block-and-retry instead of hard-fail
"""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# Maximum number of per-thread buckets to keep in memory (LRU eviction).
_MAX_THREAD_BUCKETS = 500


class RateLimitError(RuntimeError):
    """Raised when a rate limit is exceeded and retry is not configured."""


class _TokenBucket:
    """Thread-safe continuous-refill token bucket.

    Args:
        rpm: Refill rate in tokens per minute (== max burst capacity).
    """

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._tokens: float = float(rpm)  # start full
        self._last_refill: float = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time. Must be called under self._lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(float(self._rpm), self._tokens + elapsed / 60.0 * self._rpm)
        self._last_refill = now

    def consume(self) -> bool:
        """Try to consume one token.

        Returns:
            True if a token was available and consumed, False otherwise.
        """
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    @property
    def rpm(self) -> int:
        return self._rpm


class RateLimitMiddleware(AgentMiddleware[AgentState]):
    """Dual-bucket (global + per-thread) rate limiting for LLM calls.

    Instantiated once per agent factory call; the global bucket is shared via a
    class-level singleton so it accumulates across all concurrent threads.

    Args:
        global_rpm: Maximum LLM calls per minute across all threads.
            Pass 0 to disable the global bucket.
        thread_rpm: Maximum LLM calls per minute per thread_id.
            Pass 0 to disable per-thread limiting.
        retry_wait_seconds: When > 0, block-and-retry on global exhaustion
            instead of raising immediately.  Per-thread exhaustion always
            raises immediately.
    """

    # --- class-level global bucket (shared across agent instances) ---
    _global_bucket: _TokenBucket | None = None
    _global_bucket_lock = threading.Lock()

    @classmethod
    def _get_or_create_global_bucket(cls, rpm: int) -> _TokenBucket:
        with cls._global_bucket_lock:
            if cls._global_bucket is None or cls._global_bucket.rpm != rpm:
                cls._global_bucket = _TokenBucket(rpm)
            return cls._global_bucket

    @classmethod
    def reset_global_bucket(cls) -> None:
        """Reset the global bucket (useful for tests)."""
        with cls._global_bucket_lock:
            cls._global_bucket = None

    # --- instance-level per-thread buckets (LRU) ---
    def __init__(
        self,
        global_rpm: int = 60,
        thread_rpm: int = 10,
        retry_wait_seconds: float = 0.0,
    ) -> None:
        super().__init__()
        self._global_rpm = global_rpm
        self._thread_rpm = thread_rpm
        self._retry_wait = retry_wait_seconds

        # LRU store: thread_id -> _TokenBucket
        self._thread_buckets: OrderedDict[str, _TokenBucket] = OrderedDict()
        self._thread_buckets_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Bucket helpers
    # ------------------------------------------------------------------

    def _get_thread_bucket(self, thread_id: str) -> _TokenBucket:
        with self._thread_buckets_lock:
            if thread_id in self._thread_buckets:
                self._thread_buckets.move_to_end(thread_id)
            else:
                # LRU eviction
                while len(self._thread_buckets) >= _MAX_THREAD_BUCKETS:
                    evicted, _ = self._thread_buckets.popitem(last=False)
                    logger.debug("RateLimitMiddleware: evicted bucket for thread %s", evicted)
                self._thread_buckets[thread_id] = _TokenBucket(self._thread_rpm)
            return self._thread_buckets[thread_id]

    @staticmethod
    def _extract_thread_id(runtime: Runtime) -> str:
        ctx = runtime.context or {}
        return ctx.get("thread_id") or "default"

    # ------------------------------------------------------------------
    # Core check (sync) — reused by both sync and async hooks
    # ------------------------------------------------------------------

    def _check(self, runtime: Runtime) -> None:
        """Consume one token from each active bucket; raise on exhaustion."""
        thread_id = self._extract_thread_id(runtime)

        # --- per-thread check (always immediate) ---
        if self._thread_rpm > 0:
            bucket = self._get_thread_bucket(thread_id)
            if not bucket.consume():
                msg = (
                    f"Rate limit exceeded for thread '{thread_id}': "
                    f"max {self._thread_rpm} LLM calls/min per thread."
                )
                logger.warning("RateLimitMiddleware: %s", msg)
                raise RateLimitError(msg)

        # --- global check (with optional retry) ---
        if self._global_rpm > 0:
            global_bucket = self._get_or_create_global_bucket(self._global_rpm)
            if not global_bucket.consume():
                if self._retry_wait > 0:
                    logger.warning(
                        "RateLimitMiddleware: global bucket empty, waiting %.1fs before retry",
                        self._retry_wait,
                    )
                    time.sleep(self._retry_wait)
                    if not global_bucket.consume():
                        msg = (
                            f"Global rate limit exceeded: max {self._global_rpm} LLM calls/min. "
                            f"Retry after {self._retry_wait}s also failed."
                        )
                        logger.warning("RateLimitMiddleware: %s", msg)
                        raise RateLimitError(msg)
                else:
                    msg = (
                        f"Global rate limit exceeded: max {self._global_rpm} LLM calls/min."
                    )
                    logger.warning("RateLimitMiddleware: %s", msg)
                    raise RateLimitError(msg)

        logger.debug(
            "RateLimitMiddleware: allowed call for thread '%s'", thread_id
        )

    async def _acheck(self, runtime: Runtime) -> None:
        """Async version of _check; uses asyncio.sleep for the retry wait."""
        thread_id = self._extract_thread_id(runtime)

        if self._thread_rpm > 0:
            bucket = self._get_thread_bucket(thread_id)
            if not bucket.consume():
                msg = (
                    f"Rate limit exceeded for thread '{thread_id}': "
                    f"max {self._thread_rpm} LLM calls/min per thread."
                )
                logger.warning("RateLimitMiddleware: %s", msg)
                raise RateLimitError(msg)

        if self._global_rpm > 0:
            global_bucket = self._get_or_create_global_bucket(self._global_rpm)
            if not global_bucket.consume():
                if self._retry_wait > 0:
                    logger.warning(
                        "RateLimitMiddleware: global bucket empty, waiting %.1fs before retry",
                        self._retry_wait,
                    )
                    await asyncio.sleep(self._retry_wait)
                    if not global_bucket.consume():
                        msg = (
                            f"Global rate limit exceeded: max {self._global_rpm} LLM calls/min. "
                            f"Retry after {self._retry_wait}s also failed."
                        )
                        logger.warning("RateLimitMiddleware: %s", msg)
                        raise RateLimitError(msg)
                else:
                    msg = (
                        f"Global rate limit exceeded: max {self._global_rpm} LLM calls/min."
                    )
                    logger.warning("RateLimitMiddleware: %s", msg)
                    raise RateLimitError(msg)

        logger.debug(
            "RateLimitMiddleware: allowed call for thread '%s'", thread_id
        )

    # ------------------------------------------------------------------
    # AgentMiddleware hooks
    # ------------------------------------------------------------------

    @override
    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        self._check(runtime)
        return None

    @override
    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        await self._acheck(runtime)
        return None
