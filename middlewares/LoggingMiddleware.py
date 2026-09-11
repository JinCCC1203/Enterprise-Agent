from __future__ import annotations

from collections.abc import Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import ToolMessage


class LoggingMiddleware(AgentMiddleware):
    """
    Agent 日志中间件。

    支持同步和异步 Model / Tool 调用。
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[
            [ModelRequest],
            ModelResponse,
        ],
    ) -> ModelResponse:

        print("[Model] 开始调用模型")

        try:
            response = handler(request)

            print("[Model] 调用成功")

            return response

        except Exception as exc:
            print(
                f"[Model] 调用失败 | 错误: {exc}"
            )
            raise

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:

        print("[Model] 开始调用模型")

        try:
            response = await handler(request)

            print("[Model] 调用成功")

            return response

        except Exception as exc:
            print(
                f"[Model] 调用失败 | 错误: {exc}"
            )
            raise

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            ToolMessage,
        ],
    ) -> ToolMessage:

        tool_name = request.tool_call["name"]

        print(
            f"[Tool] 开始调用工具: {tool_name}"
        )

        try:
            response = handler(request)

            print(
                f"[Tool] 调用成功: {tool_name}"
            )

            return response

        except Exception as exc:
            print(
                f"[Tool] 调用失败: {tool_name} | "
                f"错误: {exc}"
            )
            raise

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage:

        tool_name = request.tool_call["name"]

        print(
            f"[Tool] 开始调用工具: {tool_name}"
        )

        try:
            response = await handler(request)

            print(
                f"[Tool] 调用成功: {tool_name}"
            )

            return response

        except Exception as exc:
            print(
                f"[Tool] 调用失败: {tool_name} | "
                f"错误: {exc}"
            )
            raise