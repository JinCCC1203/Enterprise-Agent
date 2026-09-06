from langchain.agents.middleware import ToolCallRequest,wrap_tool_call
from langchain_core.messages import ToolMessage
from typing import Callable

#为RetryMiddleware兜底，继续输出ToolMessage
@wrap_tool_call
def tool_error(
        request:ToolCallRequest,
        handler:Callable[[ToolCallRequest],ToolMessage]
) -> ToolMessage:

    try:
        return handler(request)
    except Exception as e:
        #工具执行失败
        return ToolMessage(
            content=f"工具执行失败：{str(e)}",
            tool_call_id=request.tool_call["id"]
        )