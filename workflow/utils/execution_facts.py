from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)

from workflow.state import EnterpriseAgentState


# ==============================================================
# Tool Result Extraction
# ==============================================================


def collect_tool_results(
    messages: list[BaseMessage],
) -> list[dict[str, Any]]:
    """
    从消息历史中提取 Tool 执行结果。

    注意：
        这里提取的是 Runtime Tool Execution Facts，
        不提取模型 reasoning。
    """

    results: list[dict[str, Any]] = []

    for message in messages:

        if not isinstance(
            message,
            ToolMessage,
        ):
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

        # ----------------------------------------------------------
        # Tool Name
        # ----------------------------------------------------------

        tool_name = additional_kwargs.get(
            "tool_name"
        )

        # 某些 ToolMessage 不一定包含 tool_name。
        #
        # 此时通过 tool_call_id 从前面的 AIMessage.tool_calls
        # 恢复 Tool 名称。
        if not isinstance(
            tool_name,
            str,
        ) or not tool_name.strip():

            tool_name = _find_tool_name(
                messages,
                tool_call_id,
            )

        # ----------------------------------------------------------
        # Tool Content
        # ----------------------------------------------------------

        content = message.content

        # ----------------------------------------------------------
        # Error Metadata
        # ----------------------------------------------------------

        tool_execution_error = bool(
            additional_kwargs.get(
                "tool_execution_error",
                False,
            )
        )

        error_type = additional_kwargs.get(
            "error_type"
        )

        error_message = additional_kwargs.get(
            "error_message"
        )

        # ----------------------------------------------------------
        # Structured Tool Fact
        # ----------------------------------------------------------

        results.append(
            {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "content": content,
                "error": tool_execution_error,
                "error_type": error_type,
                "error_message": error_message,
            }
        )

    return results


# ==============================================================
# Resolve Tool Name
# ==============================================================


def _find_tool_name(
    messages: list[BaseMessage],
    tool_call_id: str | None,
) -> str | None:
    """
    根据 tool_call_id 从 AIMessage.tool_calls 中寻找 Tool name。
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

        if not isinstance(
            tool_calls,
            list,
        ):
            continue

        for tool_call in tool_calls:

            if not isinstance(
                tool_call,
                dict,
            ):
                continue

            if (
                tool_call.get("id")
                != tool_call_id
            ):
                continue

            name = tool_call.get(
                "name"
            )

            if isinstance(
                name,
                str,
            ) and name.strip():

                return name

    return None


# ==============================================================
# Execution Summary
# ==============================================================


def build_execution_summary(
    state: EnterpriseAgentState,
) -> dict[str, Any]:
    """
    构造 Runtime Execution Truth。

    Finalizer 的事实来源。

    注意：
        Finalizer 不应该通过 LLM 推测：

            - Tool 是否执行
            - Tool 是否失败
            - 是否发生 HITL
            - 最近一次 HITL 决策
            - 是否发生 Recovery

        这些信息全部来自 Graph State / Tool Messages。
    """

    # ==========================================================
    # Conversation
    # ==========================================================

    messages = state.get(
        "messages",
        [],
    )

    # ==========================================================
    # Tool Results
    # ==========================================================

    tool_results = state.get(
        "tool_results",
        [],
    )

    # Specialist 如果已经把 Tool Results 写入 State，
    # 优先使用 State。
    #
    # 如果 State 中没有，则从 messages 中恢复。
    if not tool_results:

        tool_results = (
            collect_tool_results(
                messages
            )
        )

    # ==========================================================
    # Approval Events
    # ==========================================================

    approval_events = state.get(
        "approval_events",
        [],
    )

    # ==========================================================
    # Runtime Execution Truth
    # ==========================================================

    return {
        # ------------------------------------------------------
        # Workflow
        # ------------------------------------------------------

        "task_status": state.get(
            "task_status"
        ),

        "current_agent": state.get(
            "current_agent"
        ),

        "next_agent": state.get(
            "next_agent"
        ),

        "handoff_reason": state.get(
            "handoff_reason"
        ),

        # ------------------------------------------------------
        # Tool-level HITL
        # ------------------------------------------------------

        "approval_required": bool(
            state.get(
                "approval_required",
                False,
            )
        ),

        # 最近一次 Tool-level HITL 决策
        "approval_status": state.get(
            "approval_status"
        ),

        # 整个 Workflow 的 Tool Approval 历史
        "approval_events": approval_events,

        # ------------------------------------------------------
        # Recovery
        # ------------------------------------------------------

        "recovery_status": state.get(
            "recovery_status"
        ),

        "recovery_attempts": state.get(
            "recovery_attempts",
            0,
        ),

        "recovery_reason": state.get(
            "recovery_reason"
        ),

        "resume_required": bool(
            state.get(
                "resume_required",
                False,
            )
        ),

        # ------------------------------------------------------
        # Error / Failure
        # ------------------------------------------------------

        "error": state.get(
            "error"
        ),

        "last_failed_node": state.get(
            "last_failed_node"
        ),

        "last_failed_tool": state.get(
            "last_failed_tool"
        ),

        # ------------------------------------------------------
        # Tool Execution
        # ------------------------------------------------------

        "tool_results": tool_results,
    }