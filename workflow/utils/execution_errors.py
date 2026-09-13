from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage


def extract_tool_execution_error(
    messages: list[BaseMessage],
) -> dict[str, Any] | None:
    """
    从 Agent 输出消息中提取最终的 Tool Execution Error。

    ToolErrorMiddleware 在 Tool 最终失败后，会通过
    ToolMessage.additional_kwargs 写入结构化错误信息：

        tool_execution_error = True
        retry_exhausted = True
        tool_name
        error_type
        error_message

    本函数负责将这些信息转换成 Workflow 层可以消费的
    结构化 failure。

    返回：
        {
            "error": "...",
            "last_failed_tool": "...",
            "error_type": "...",
            "retry_exhausted": True,
        }

    如果当前消息列表中不存在 Tool Execution Error：
        return None
    """

    for message in reversed(messages):

        # ----------------------------------------------------------
        # 只处理 ToolMessage
        # ----------------------------------------------------------

        if not isinstance(message, ToolMessage):
            continue

        # ----------------------------------------------------------
        # additional_kwargs
        #
        # LangChain 的 additional_kwargs 理论上是 dict，
        # 这里仍做类型保护，避免后续自定义消息结构导致异常。
        # ----------------------------------------------------------

        metadata = message.additional_kwargs

        if not isinstance(metadata, dict):
            continue

        # ----------------------------------------------------------
        # 只识别 ToolErrorMiddleware 标记过的错误消息
        # ----------------------------------------------------------

        if not metadata.get(
            "tool_execution_error",
            False,
        ):
            continue

        # ----------------------------------------------------------
        # Error Message
        # ----------------------------------------------------------

        error_message = metadata.get(
            "error_message",
        )

        if not isinstance(
            error_message,
            str,
        ) or not error_message.strip():

            content = message.content

            if isinstance(
                content,
                str,
            ):
                error_message = content

            else:
                error_message = str(
                    content
                )

        # ----------------------------------------------------------
        # Tool Name
        # ----------------------------------------------------------

        tool_name = metadata.get(
            "tool_name",
        )

        if not isinstance(
            tool_name,
            str,
        ) or not tool_name.strip():

            tool_name = None

        # ----------------------------------------------------------
        # Error Type
        # ----------------------------------------------------------

        error_type = metadata.get(
            "error_type",
        )

        if not isinstance(
            error_type,
            str,
        ) or not error_type.strip():

            error_type = None

        # ----------------------------------------------------------
        # Retry Exhausted
        # ----------------------------------------------------------

        retry_exhausted = metadata.get(
            "retry_exhausted",
            False,
        )

        retry_exhausted = bool(
            retry_exhausted
        )

        # ----------------------------------------------------------
        # 返回 Workflow Failure
        # ----------------------------------------------------------

        return {
            "error": error_message,
            "last_failed_tool": tool_name,
            "error_type": error_type,
            "retry_exhausted": retry_exhausted,
        }

    return None