"""LoggingMiddleware - 演示中间件的所有钩子方法"""

import logging
import time
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from deerflow.agents.thread_state import ThreadState

logger = logging.getLogger(__name__)


class LoggingMiddleware(AgentMiddleware[ThreadState]):
    state_schema = ThreadState

    def __init__(self, prefix: str = "🔍"):
        super().__init__()
        self._prefix = prefix
        self._start_time: float | None = None

    @override
    def before_agent(self, state: ThreadState, runtime: Runtime) -> dict | None:
        self._start_time = time.time()

        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None

        logger.info(f"{self._prefix} ═══════════════════════════════════════════")
        logger.info(f"{self._prefix} BEFORE_AGENT")
        logger.info(f"{self._prefix} ───────────────────────────────────────────")
        logger.info(f"{self._prefix} Messages count: {len(messages)}")
        logger.info(f"{self._prefix} Last message type: {last_msg.type if last_msg else 'None'}")
        logger.info(f"{self._prefix} Thread title: {state.get('title', 'None')}")
        logger.info(f"{self._prefix} Sandbox: {state.get('sandbox', {}).get('sandbox_id', 'None') if state.get('sandbox') else 'None'}")

        return None

    @override
    def after_agent(self, state: ThreadState, runtime: Runtime) -> dict | None:
        elapsed = time.time() - self._start_time if self._start_time else 0

        messages = state.get("messages", [])
        artifacts = state.get("artifacts", [])

        logger.info(f"{self._prefix} ───────────────────────────────────────────")
        logger.info(f"{self._prefix} AFTER_AGENT")
        logger.info(f"{self._prefix} ───────────────────────────────────────────")
        logger.info(f"{self._prefix} Total messages: {len(messages)}")
        logger.info(f"{self._prefix} Artifacts: {artifacts}")
        logger.info(f"{self._prefix} Elapsed: {elapsed:.2f}s")
        logger.info(f"{self._prefix} ═══════════════════════════════════════════")

        return None

    # =========================================================================
    # Model 调用钩子
    # =========================================================================

    @override
    def before_model(self, state: ThreadState, runtime: Runtime) -> dict | None:
        """LLM 调用前"""
        messages = state.get("messages", [])
        logger.info(f"{self._prefix} 📤 BEFORE_MODEL - Sending {len(messages)} messages to LLM")
        return None

    @override
    def after_model(self, state: ThreadState, runtime: Runtime) -> dict | None:
        """LLM 调用后"""
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None

        if last_msg and last_msg.type == "ai":
            tool_calls = getattr(last_msg, "tool_calls", [])
            content_preview = str(last_msg.content)[:100] if last_msg.content else ""

            logger.info(f"{self._prefix} 📥 AFTER_MODEL")
            logger.info(f"{self._prefix}    Content: {content_preview}...")
            logger.info(f"{self._prefix}    Tool calls: {len(tool_calls)}")

            for tc in tool_calls:
                logger.info(f"{self._prefix}    → {tc.get('name')}({tc.get('args', {})})")

        return None

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """包装 LLM 调用（同步）- 可以修改请求/响应"""
        logger.info(f"{self._prefix} 🔄 WRAP_MODEL_CALL (sync)")
        logger.info(f"{self._prefix}    Input messages: {len(request.messages)}")

        start = time.time()
        result = handler(request)  # 调用实际的 LLM
        elapsed = time.time() - start

        logger.info(f"{self._prefix}    LLM call took: {elapsed:.2f}s")
        return result

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """包装 LLM 调用（异步）"""
        logger.info(f"{self._prefix} 🔄 AWRAP_MODEL_CALL (async)")

        start = time.time()
        result = await handler(request)
        elapsed = time.time() - start

        logger.info(f"{self._prefix}    LLM call took: {elapsed:.2f}s")
        return result

    # =========================================================================
    # Tool 调用钩子
    # =========================================================================

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """包装工具调用（同步）"""
        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})

        logger.info(f"{self._prefix} 🔧 WRAP_TOOL_CALL: {tool_name}")
        logger.info(f"{self._prefix}    Args: {tool_args}")

        start = time.time()
        result = handler(request)  # 调用实际的工具
        elapsed = time.time() - start

        if isinstance(result, ToolMessage):
            content_preview = str(result.content)[:100]
            logger.info(f"{self._prefix}    Result: {content_preview}...")
            logger.info(f"{self._prefix}    Status: {result.status}")

        logger.info(f"{self._prefix}    Tool took: {elapsed:.2f}s")
        return result

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """包装工具调用（异步）"""
        tool_name = request.tool_call.get("name", "unknown")

        logger.info(f"{self._prefix} 🔧 AWRAP_TOOL_CALL: {tool_name}")

        start = time.time()
        result = await handler(request)
        elapsed = time.time() - start

        logger.info(f"{self._prefix}    Tool took: {elapsed:.2f}s")
        return result
