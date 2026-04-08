from pydantic import BaseModel, Field


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting.

    Two independent token buckets protect the LLM layer:
    - global_rpm: total model invocations allowed per minute across all threads.
    - thread_rpm: per-thread model invocations allowed per minute.

    When a limit is hit the middleware raises RateLimitError which surfaces to
    the caller as a ToolMessage / error event rather than crashing the run.
    """

    enabled: bool = Field(default=False, description="Enable rate limiting middleware")
    global_rpm: int = Field(default=60, description="Max LLM calls per minute across all threads (0 = unlimited)")
    thread_rpm: int = Field(default=10, description="Max LLM calls per minute per thread (0 = unlimited)")
    # How long (seconds) to wait before retrying when the global bucket is empty.
    # A value > 0 makes the middleware block-and-retry instead of immediately raising.
    retry_wait_seconds: float = Field(default=0.0, description="Seconds to wait and retry when globally rate-limited (0 = raise immediately)")
