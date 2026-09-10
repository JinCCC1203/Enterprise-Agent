from __future__ import annotations

from collections.abc import Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

from tools_manager.tool_exposure import PermissionBasedToolExposure


class DynamicToolMiddleware(AgentMiddleware):
    """
    在每次 Model Call 前，根据当前运行时用户角色，动态决定哪些 Tool 可以暴露给模型。

    注意：
        DynamicToolMiddleware 不负责决定LLM 最终调用哪个 Tool。
        它只负责在 Model Call 前控制Tool Exposure。
    """

    def __init__(
        self,
        tool_exposure: PermissionBasedToolExposure,
    ) -> None:
        super().__init__()

        self.tool_exposure = tool_exposure

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[
            [ModelRequest],
            ModelResponse,
        ],
    ) -> ModelResponse:

        # 获取 Runtime Context
        context = request.runtime.context

        # 获取当前用户角色
        user_role = context.user_role

        # 根据权限决定哪些 Tool 可以暴露给 LLM
        exposed_tools = self.tool_exposure.expose(
            role=user_role,
        )

        # 只向当前 Model Call 暴露允许使用的 Tool
        new_request = request.override(
            tools=exposed_tools,
        )

        return handler(new_request)

    @staticmethod
    def _extract_query(
        request: ModelRequest,
    ) -> str:
        """
        获取当前请求中的最后一条用户消息。在目前的动态工具选择中间件中用不上。
        后续可以根据PermissionBasedToolExposure的扩展，来进一步决定是否使用用户消息。
        """
        for message in reversed(request.messages):

            message_type = getattr(
                message,
                "type",
                None,
            )

            if message_type in {"human", "user"}:

                content = getattr(
                    message,
                    "content",
                    "",
                )

                if isinstance(content, str):
                    return content

        return ""