from __future__ import annotations

from collections.abc import Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

from tools_manager.selectors import (
    RuleBasedToolSelector,
    SelectionContext,
)


class DynamicToolMiddleware(AgentMiddleware):
    """
    在每次 Model Call 前，
    动态决定哪些 Tool 暴露给模型。

    流程：

        Runtime Context
              ↓
        SelectionContext
              ↓
        RuleBasedToolSelector
              ↓
        candidate tools
              ↓
        request.override(tools=...)
              ↓
        LLM
    """

    def __init__(
        self,
        selector: RuleBasedToolSelector,
    ) -> None:

        super().__init__()

        self.selector = selector

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

        # 构造 Selector Context
        selection_context = SelectionContext(
            role=context.user_role,
            task_type=context.task_type,
            query=self._extract_query(request),
        )

        # Selector 筛选候选 Tool
        selected_tools = self.selector.select(
            selection_context
        )

        # 只把候选 Tool 暴露给当前 Model Call
        new_request = request.override(
            tools=selected_tools
        )

        return handler(new_request)

    @staticmethod
    def _extract_query(
        request: ModelRequest,
    ) -> str:
        """
        获取当前请求中的最后一条用户消息。
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