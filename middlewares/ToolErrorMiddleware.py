from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import (
    ToolCallRequest,
    wrap_tool_call,
)
from langchain_core.messages import ToolMessage


# ==============================================================
# Helpers
# ==============================================================


def _build_tool_error_message(
    *,
    request: ToolCallRequest,
    error: Exception,
) -> ToolMessage:
    """
    构造结构化 Tool Error Message。

    ToolErrorMiddleware 是：
        Retry exhausted
            ↓
        Error normalization
            ↓
        Structured ToolMessage
    """

    tool_call = request.tool_call

    tool_name = tool_call.get(
        "name",
        "unknown",
    )

    tool_call_id = tool_call.get(
        "id",
        "",
    )

    error_type = type(error).__name__

    message = (
        f"工具执行失败：{str(error)}"
    )

    return ToolMessage(
        content=message,
        tool_call_id=tool_call_id,
        additional_kwargs={
            "tool_execution_error": True,
            "retry_exhausted": True,
            "tool_name": tool_name,
            "error_type": error_type,
            "error_message": str(error),
        },
    )


# ==============================================================
# Sync Tool Error
# ==============================================================


@wrap_tool_call
def tool_error(
    request: ToolCallRequest,
    handler: Callable[
        [ToolCallRequest],
        ToolMessage,
    ],
) -> ToolMessage:
    """
    同步 Tool Error Middleware。

    只有当内层 Tool Retry 已经耗尽，
    仍然抛出异常时，这里才负责捕获。
    """

    try:

        return handler(request)

    except Exception as exc:

        return _build_tool_error_message(
            request=request,
            error=exc,
        )


# ==============================================================
# Async Tool Error
# ==============================================================


@wrap_tool_call
async def tool_error_async(
    request: ToolCallRequest,
    handler: Callable[
        [ToolCallRequest],
        ToolMessage,
    ],
) -> ToolMessage:
    """
    异步 Tool Error Middleware。

    调用链：
        Tool
          ↓
        RetryMiddleware
          ↓
        retry × 3
          ↓
        still failed
          ↓
        ToolErrorMiddleware
          ↓
        Structured ToolMessage
    """

    try:

        return await handler(request)

    except Exception as exc:

        return _build_tool_error_message(
            request=request,
            error=exc,
        )