from pydantic import BaseModel, Field


class TokenUsageConfig(BaseModel):
    """Configuration for token usage tracking and circuit-breaker protection.

    Two layers of protection:
      1. Logging — always records input/output/total tokens after each model call.
      2. Circuit breaker — optional hard caps on cumulative token spend per thread.
         When a thread exceeds the warn threshold a reminder is injected into the
         conversation; when it exceeds the hard limit tool_calls are stripped so
         the model is forced to produce a final answer instead of looping further.

    All token thresholds default to 0, which means "disabled".
    """

    enabled: bool = Field(default=False, description="Enable token usage tracking middleware")

    # ── Circuit-breaker thresholds (0 = disabled) ──────────────────────────
    max_output_tokens_per_thread: int = Field(
        default=0,
        description="Hard limit on cumulative output tokens per thread (0 = unlimited)",
    )
    max_total_tokens_per_thread: int = Field(
        default=0,
        description="Hard limit on cumulative total tokens per thread (0 = unlimited)",
    )
    warn_threshold_ratio: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Fraction of the hard limit at which a budget-warning is injected (0.8 = 80%%)",
    )

    # ── LRU cap on per-thread state ─────────────────────────────────────────
    max_tracked_threads: int = Field(
        default=200,
        description="Maximum number of threads whose token budgets are kept in memory (LRU eviction)",
    )
