from __future__ import annotations

from collections.abc import Callable

from langchain.agents.middleware import (
    ToolCallRequest,
    wrap_tool_call,
)
from langchain_core.messages import ToolMessage


@wrap_tool_call
def tool_error(
    request: ToolCallRequest,
    handler: Callable[
        [ToolCallRequest],
        ToolMessage,
    ],
) -> ToolMessage:

    try:
        return handler(request)

    except Exception as e:
        return ToolMessage(
            content=f"工具执行失败：{str(e)}",
            tool_call_id=request.tool_call["id"],
        )


@wrap_tool_call
async def tool_error_async(
    request: ToolCallRequest,
    handler,
) -> ToolMessage:

    try:
        return await handler(request)

    except Exception as e:
        return ToolMessage(
            content=f"工具执行失败：{str(e)}",
            tool_call_id=request.tool_call["id"],
        )