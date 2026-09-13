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

    Retry Exhausted 后：

        RetryMiddleware
            ↓
        ToolErrorMiddleware
            ↓
        构造结构化错误事实
            ↓
        继续向上抛出异常
            ↓
        Specialist Node
            ↓
        Recovery
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


def _annotate_tool_error(
    *,
    request: ToolCallRequest,
    error: Exception,
) -> None:
    """
    将 Tool Error Runtime Facts 附着到异常对象。

    这样 Specialist Node 在捕获异常时，
    即使没有 ToolMessage，也可以知道：

        - 哪个 Tool 失败
        - 是否 Retry Exhausted
        - 错误类型
        - 错误信息
    """

    tool_call = request.tool_call

    tool_name = tool_call.get(
        "name"
    )

    tool_call_id = tool_call.get(
        "id"
    )

    setattr(
        error,
        "tool_execution_error",
        True,
    )

    setattr(
        error,
        "retry_exhausted",
        True,
    )

    setattr(
        error,
        "tool_name",
        tool_name,
    )

    setattr(
        error,
        "tool_call_id",
        tool_call_id,
    )

    setattr(
        error,
        "error_type",
        type(error).__name__,
    )

    setattr(
        error,
        "error_message",
        str(error),
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

    调用链：

        Tool
          ↓
        Retry
          ↓
        Retry Exhausted
          ↓
        ToolError
          ↓
        annotate error
          ↓
        raise
          ↓
        Specialist
          ↓
        Recovery
    """

    try:

        return handler(request)

    except Exception as exc:

        _annotate_tool_error(
            request=request,
            error=exc,
        )

        # 构造 ToolMessage，保留统一错误语义。
        #
        # 当前架构不直接 return，
        # 因为 return 会让 Agent 继续运行。
        _build_tool_error_message(
            request=request,
            error=exc,
        )

        # 关键：
        # 不能吞掉异常。
        raise


# ==============================================================
# Async Tool Error
# ==============================================================


@wrap_tool_call
async def tool_error_async(
    request: ToolCallRequest,
    handler: Callable[
        [ToolCallRequest],
        Any,
    ],
) -> Any:
    """
    异步 Tool Error Middleware。

    Retry Exhausted 后：

        不返回 ToolMessage
        不继续 Agent Loop
        直接 raise
    """

    try:

        return await handler(request)

    except Exception as exc:

        _annotate_tool_error(
            request=request,
            error=exc,
        )

        # 构造结构化错误事实。
        _build_tool_error_message(
            request=request,
            error=exc,
        )

        # 关键：
        # 将异常继续向 Specialist Node 传播。
        raise