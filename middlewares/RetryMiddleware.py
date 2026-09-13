from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import ToolMessage


# ==============================================================
# Model Retry
# ==============================================================


@wrap_model_call
async def retry_model_async(
    request: ModelRequest,
    handler: Callable[
        [ModelRequest],
        ModelResponse,
    ],
) -> ModelResponse:
    """
    Model-level retry middleware。

    职责：
        - 捕获 Model Call transient failure
        - 自动重试
        - 达到最大次数后继续抛出异常

    注意：
        这里不修改 LangGraph State。

        Model retry 属于 Agent Runtime execution layer。
    """

    max_attempts = 3

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):

        try:
            return await handler(request)

        except Exception as exc:

            last_error = exc

            print(
                "[Model Retry] "
                f"attempt={attempt}/{max_attempts} "
                f"error={exc}"
            )

            # ------------------------------------------------------
            # Retry exhausted
            # ------------------------------------------------------

            if attempt >= max_attempts:
                raise

    # 理论上不会执行到这里，
    # 仅用于类型安全。
    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "Model retry exited without a result."
    )


# ==============================================================
# Tool Retry
# ==============================================================


@wrap_tool_call
async def retry_tool_async(
    request: ToolCallRequest,
    handler: Callable[
        [ToolCallRequest],
        ToolMessage,
    ],
) -> ToolMessage:
    """
    Tool-level retry middleware。

    职责：
        - 捕获 Tool execution failure
        - 自动重试
        - 重试耗尽后继续抛出异常

    注意：
        ToolErrorMiddleware 必须位于该 Middleware 的外层，
        才能在 retry exhausted 后捕获最终异常。
    """

    max_attempts = 3

    tool_name = (
        request.tool_call.get(
            "name",
            "unknown",
        )
    )

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):

        try:

            print(
                "[Tool Retry] "
                f"tool={tool_name} "
                f"attempt={attempt}/{max_attempts}"
            )

            return await handler(request)

        except Exception as exc:

            last_error = exc

            print(
                "[Tool Retry Error] "
                f"tool={tool_name} "
                f"attempt={attempt}/{max_attempts} "
                f"error={exc}"
            )

            # ------------------------------------------------------
            # Retry exhausted
            # ------------------------------------------------------

            if attempt >= max_attempts:
                raise

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "Tool retry exited without a result."
    )