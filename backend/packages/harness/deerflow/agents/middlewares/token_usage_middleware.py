"""Token-usage tracking + circuit-breaker middleware.

Two responsibilities:
  1. **Logging** — records input/output/total tokens after every model call.
  2. **Circuit breaker** — enforces per-thread cumulative token budgets.

Circuit-breaker behaviour (mirrors LoopDetectionMiddleware):
  - Tracks cumulative output_tokens and total_tokens per thread_id in an
    LRU-evicting OrderedDict (bounded by ``max_tracked_threads``).
  - warn threshold  (ratio × hard limit): injects a HumanMessage warning so the
    model knows it is approaching its budget and should wrap up.  Warning is
    injected only once per threshold crossing to avoid message spam.
  - hard limit: strips tool_calls from the last AIMessage, forcing the model to
    produce a final text answer instead of issuing more tool calls.

Configuration (config.yaml → token_usage):
    token_usage:
      enabled: true
      max_output_tokens_per_thread: 50000   # 0 = unlimited
      max_total_tokens_per_thread:  200000  # 0 = unlimited
      warn_threshold_ratio: 0.8
      max_tracked_threads: 200
"""

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from deerflow.config.token_usage_config import TokenUsageConfig

logger = logging.getLogger(__name__)

_WARN_MSG = (
    "[TOKEN BUDGET WARNING] You are approaching the token limit for this conversation. "
    "Please wrap up your current task and provide a final answer soon."
)
_HARD_STOP_MSG = (
    "[TOKEN BUDGET EXCEEDED] The token budget for this conversation has been exhausted. "
    "Produce your final answer now using only the information already gathered."
)


@dataclass
class _ThreadBudget:
    """Cumulative token counters for a single thread."""

    output_tokens: int = 0
    total_tokens: int = 0
    # Track which hard-limit dimensions have already had a warning injected
    # so we only warn once per threshold crossing.
    warned: set[str] = field(default_factory=set)


class TokenUsageMiddleware(AgentMiddleware):
    """Logs LLM token usage and enforces per-thread token budgets (circuit breaker).

    Args:
        config: TokenUsageConfig instance.  When *None* the middleware loads the
                config from ``get_app_config()`` on first use.
    """

    def __init__(self, config: TokenUsageConfig | None = None) -> None:
        super().__init__()
        self._config = config
        self._lock = threading.Lock()
        # LRU: thread_id -> _ThreadBudget
        self._budgets: OrderedDict[str, _ThreadBudget] = OrderedDict()

    # ── config helper ──────────────────────────────────────────────────────

    def _get_config(self) -> TokenUsageConfig:
        if self._config is not None:
            return self._config
        try:
            from deerflow.config.app_config import get_app_config
            return get_app_config().token_usage
        except Exception:
            return TokenUsageConfig()

    # ── thread-budget helpers ──────────────────────────────────────────────

    @staticmethod
    def _extract_thread_id(runtime: Runtime) -> str:
        ctx = runtime.context or {}
        return ctx.get("thread_id") or "default"

    def _get_budget(self, thread_id: str, max_tracked: int) -> _ThreadBudget:
        """Return (and LRU-touch) the budget for *thread_id*, creating if absent."""
        with self._lock:
            if thread_id in self._budgets:
                self._budgets.move_to_end(thread_id)
            else:
                # Evict oldest entries when over the cap
                while len(self._budgets) >= max_tracked:
                    evicted, _ = self._budgets.popitem(last=False)
                    logger.debug("TokenUsageMiddleware: evicted budget for thread %s", evicted)
                self._budgets[thread_id] = _ThreadBudget()
            return self._budgets[thread_id]

    def _accumulate(self, budget: _ThreadBudget, usage: dict) -> None:
        with self._lock:
            budget.output_tokens += usage.get("output_tokens", 0)
            budget.total_tokens += usage.get("total_tokens", 0)

    # ── core logic ─────────────────────────────────────────────────────────

    def _process(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Log usage and apply circuit-breaker checks. Returns state patch or None."""
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if getattr(last, "type", None) != "ai":
            return None

        usage = getattr(last, "usage_metadata", None)

        # ── 1. Logging ─────────────────────────────────────────────────────
        if usage:
            logger.info(
                "LLM token usage: input=%s output=%s total=%s",
                usage.get("input_tokens", "?"),
                usage.get("output_tokens", "?"),
                usage.get("total_tokens", "?"),
            )

        # ── 2. Circuit breaker ─────────────────────────────────────────────
        cfg = self._get_config()
        # No limits configured → logging only, nothing more to do.
        if not cfg.max_output_tokens_per_thread and not cfg.max_total_tokens_per_thread:
            return None
        if not usage:
            return None

        thread_id = self._extract_thread_id(runtime)
        budget = self._get_budget(thread_id, cfg.max_tracked_threads)
        self._accumulate(budget, usage)

        with self._lock:
            out_tok = budget.output_tokens
            tot_tok = budget.total_tokens
            warned = budget.warned

        # Check each active limit dimension
        for dim, current, limit in (
            ("output", out_tok, cfg.max_output_tokens_per_thread),
            ("total", tot_tok, cfg.max_total_tokens_per_thread),
        ):
            if not limit:
                continue  # dimension disabled

            warn_at = int(limit * cfg.warn_threshold_ratio)

            # Hard limit hit → strip tool_calls, force final answer
            if current >= limit:
                logger.error(
                    "TokenUsageMiddleware: hard limit reached for thread %s "
                    "(%s_tokens=%d >= limit=%d) — forcing stop",
                    thread_id, dim, current, limit,
                )
                tool_calls = getattr(last, "tool_calls", None)
                if tool_calls:
                    stripped = last.model_copy(
                        update={
                            "tool_calls": [],
                            "content": (last.content or "") + f"\n\n{_HARD_STOP_MSG}",
                        }
                    )
                    return {"messages": [stripped]}
                # No tool_calls to strip (already a final answer) — nothing to do
                return None

            # Warn threshold crossed, warn only once per dimension per thread
            warn_key = f"warn_{dim}"
            if current >= warn_at and warn_key not in warned:
                with self._lock:
                    budget.warned.add(warn_key)
                logger.warning(
                    "TokenUsageMiddleware: warn threshold reached for thread %s "
                    "(%s_tokens=%d >= warn_at=%d, limit=%d)",
                    thread_id, dim, current, warn_at, limit,
                )
                return {"messages": [HumanMessage(content=_WARN_MSG)]}

        return None

    # ── AgentMiddleware hooks ──────────────────────────────────────────────

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._process(state, runtime)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._process(state, runtime)

    # ── Test helpers ───────────────────────────────────────────────────────

    def reset(self, thread_id: str | None = None) -> None:
        """Clear budget state. Pass thread_id to reset one thread, None for all."""
        with self._lock:
            if thread_id:
                self._budgets.pop(thread_id, None)
            else:
                self._budgets.clear()

    def get_budget_snapshot(self, thread_id: str) -> dict:
        """Return a copy of the current budget counters for *thread_id* (for tests)."""
        with self._lock:
            b = self._budgets.get(thread_id)
            if b is None:
                return {"output_tokens": 0, "total_tokens": 0}
            return {"output_tokens": b.output_tokens, "total_tokens": b.total_tokens}
