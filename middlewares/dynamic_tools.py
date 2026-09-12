from __future__ import annotations

from collections.abc import Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

from tools_manager.tool_exposure import (
    PermissionBasedToolExposure,
)


class DynamicToolMiddleware(AgentMiddleware):
    """
    在每次 Model Call 前，
    根据当前 Runtime Role，
    对当前 Specialist Scope 内的 Tool
    做权限过滤。
    """

    def __init__(
        self,
        *,
        tool_exposure: PermissionBasedToolExposure,
    ) -> None:
        super().__init__()

        self.tool_exposure = tool_exposure

    def _get_exposed_tools(
        self,
        request: ModelRequest,
    ):
        role = request.runtime.context.user_role

        return self.tool_exposure.expose(
            role=role,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[
            [ModelRequest],
            ModelResponse,
        ],
    ) -> ModelResponse:

        exposed_tools = self._get_exposed_tools(
            request
        )

        new_request = request.override(
            tools=exposed_tools,
        )

        return handler(new_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:

        exposed_tools = self._get_exposed_tools(
            request
        )

        new_request = request.override(
            tools=exposed_tools,
        )

        return await handler(new_request)

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