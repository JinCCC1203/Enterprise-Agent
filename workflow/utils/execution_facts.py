from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)

from workflow.state import EnterpriseAgentState


def collect_tool_results(
    messages: list[BaseMessage],
) -> list[dict[str, Any]]:
    """
    从消息历史中提取 Tool 执行结果。

    注意：
        这里提取的是 Tool execution facts，
        不提取模型 reasoning。
    """

    results: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        tool_call_id = getattr(
            message,
            "tool_call_id",
            None,
        )

        additional_kwargs = (
            message.additional_kwargs
            if isinstance(
                message.additional_kwargs,
                dict,
            )
            else {}
        )

        tool_name = additional_kwargs.get(
            "tool_name"
        )

        # 某些 ToolMessage 不一定包含 tool_name，
        # 可以从前面的 AIMessage tool_call 中补充。
        if not tool_name:
            tool_name = _find_tool_name(
                messages,
                tool_call_id,
            )

        content = message.content

        results.append(
            {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "content": content,
                "error": bool(
                    additional_kwargs.get(
                        "tool_execution_error",
                        False,
                    )
                ),
                "error_type": additional_kwargs.get(
                    "error_type"
                ),
                "error_message": additional_kwargs.get(
                    "error_message"
                ),
            }
        )

    return results


def _find_tool_name(
    messages: list[BaseMessage],
    tool_call_id: str | None,
) -> str | None:
    """
    从对应 AIMessage.tool_calls 中寻找 Tool name。
    """

    if not tool_call_id:
        return None

    for message in messages:
        if not isinstance(
            message,
            AIMessage,
        ):
            continue

        tool_calls = getattr(
            message,
            "tool_calls",
            [],
        )

        for tool_call in tool_calls:

            if not isinstance(
                tool_call,
                dict,
            ):
                continue

            if (
                tool_call.get("id")
                == tool_call_id
            ):
                name = tool_call.get(
                    "name"
                )

                if isinstance(
                    name,
                    str,
                ):
                    return name

    return None


def build_execution_summary(
    state: EnterpriseAgentState,
) -> dict[str, Any]:
    """
    构造 Runtime Execution Truth。

    这是 Finalizer 唯一可信的事实来源。
    """

    messages = state.get(
        "messages",
        [],
    )

    tool_results = state.get(
        "tool_results",
        [],
    )

    if not tool_results:
        tool_results = collect_tool_results(
            messages
        )

    return {
        "task_status": state.get(
            "task_status"
        ),
        "current_agent": state.get(
            "current_agent"
        ),
        "approval_required": bool(
            state.get(
                "approval_required",
                False,
            )
        ),
        "approval_status": state.get(
            "approval_status"
        ),
        "recovery_status": state.get(
            "recovery_status"
        ),
        "recovery_attempts": state.get(
            "recovery_attempts",
            0,
        ),
        "error": state.get(
            "error"
        ),
        "last_failed_node": state.get(
            "last_failed_node"
        ),
        "last_failed_tool": state.get(
            "last_failed_tool"
        ),
        "tool_results": tool_results,
    }