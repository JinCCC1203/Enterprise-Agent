from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)

from workflow.state import EnterpriseAgentState
from workflow.utils.execution_facts import (
    build_execution_summary,
)


def create_finalizer_node():
    """
    创建 Workflow Finalizer。

    Finalizer 的职责：

        Runtime State
            +
        Tool Execution Facts
            ↓
        最终用户可见答案

    Finalizer 不：
        - 调用 Tool
        - 修改业务数据
        - 再次进行 Agent Routing
        - 推测 Runtime State
    """

    async def finalizer_node(
        state: EnterpriseAgentState,
    ) -> dict[str, Any]:

        execution_summary = (
            build_execution_summary(
                state
            )
        )

        # ----------------------------------------------------------
        # 1. Tool / Workflow Execution Facts
        # ----------------------------------------------------------

        task_status = execution_summary.get(
            "task_status"
        )

        approval_required = (
            execution_summary.get(
                "approval_required",
                False,
            )
        )

        approval_status = (
            execution_summary.get(
                "approval_status"
            )
        )

        recovery_status = (
            execution_summary.get(
                "recovery_status"
            )
        )

        error = execution_summary.get(
            "error"
        )

        tool_results = (
            execution_summary.get(
                "tool_results",
                [],
            )
        )

        # ----------------------------------------------------------
        # 2. Failure
        # ----------------------------------------------------------

        if task_status == "failed":

            answer = _build_failure_answer(
                execution_summary
            )

            return {
                "execution_summary":
                    execution_summary,
                "final_answer": answer,
            }

        # ----------------------------------------------------------
        # 3. Tool-based execution
        # ----------------------------------------------------------

        if tool_results:

            answer = _build_tool_answer(
                state=state,
                execution_summary=
                    execution_summary,
            )

            return {
                "execution_summary":
                    execution_summary,
                "final_answer": answer,
            }

        # ----------------------------------------------------------
        # 4. 普通 Agent 问答
        # ----------------------------------------------------------

        answer = _extract_latest_ai_message(
            state.get(
                "messages",
                [],
            )
        )

        if not answer:
            answer = (
                "Workflow completed successfully, "
                "but no final answer was generated."
            )

        return {
            "execution_summary":
                execution_summary,
            "final_answer": answer,
        }

    return finalizer_node


def _build_tool_answer(
    *,
    state: EnterpriseAgentState,
    execution_summary: dict[str, Any],
) -> str:
    """
    构造 Tool 执行类任务的最终答案。

    核心原则：
        关键事实全部来自 Runtime State，
        而不是来自 LLM 自己的猜测。
    """

    lines: list[str] = []

    task_status = execution_summary.get(
        "task_status"
    )

    approval_required = execution_summary.get(
        "approval_required",
        False,
    )

    approval_status = execution_summary.get(
        "approval_status"
    )

    recovery_status = execution_summary.get(
        "recovery_status"
    )

    tool_results = execution_summary.get(
        "tool_results",
        [],
    )

    # --------------------------------------------------------------
    # Workflow status
    # --------------------------------------------------------------

    if task_status == "completed":
        lines.append(
            "✅ Workflow 已成功完成。"
        )
    else:
        lines.append(
            f"Workflow 状态：{task_status}"
        )

    # --------------------------------------------------------------
    # HITL
    # --------------------------------------------------------------

    if approval_required:

        if approval_status == "approved":
            lines.append(
                "人工审批：已通过。"
            )

        elif approval_status == "edited":
            lines.append(
                "人工审批：已通过，并使用人工修改后的参数执行。"
            )

        elif approval_status == "rejected":
            lines.append(
                "人工审批：已拒绝。"
            )

        elif approval_status == "pending":
            lines.append(
                "人工审批：等待审批。"
            )

    # --------------------------------------------------------------
    # Recovery
    # --------------------------------------------------------------

    if recovery_status:
        lines.append(
            f"Workflow Recovery：{recovery_status}"
        )

    # --------------------------------------------------------------
    # Tool Results
    # --------------------------------------------------------------

    successful_tools: list[dict[str, Any]] = []

    failed_tools: list[dict[str, Any]] = []

    for result in tool_results:

        if result.get("error"):
            failed_tools.append(result)
        else:
            successful_tools.append(result)

    if successful_tools:
        lines.append(
            "\n已成功执行的工具："
        )

        for result in successful_tools:
            tool_name = result.get(
                "tool_name"
            )

            content = result.get(
                "content"
            )

            lines.append(
                f"- {tool_name}: {content}"
            )

    if failed_tools:
        lines.append(
            "\n工具执行失败："
        )

        for result in failed_tools:

            tool_name = result.get(
                "tool_name"
            )

            error_message = result.get(
                "error_message"
            )

            lines.append(
                f"- {tool_name}: "
                f"{error_message}"
            )

    return "\n".join(lines)


def _build_failure_answer(
    execution_summary: dict[str, Any],
) -> str:

    error = execution_summary.get(
        "error"
    )

    failed_tool = execution_summary.get(
        "last_failed_tool"
    )

    recovery_status = execution_summary.get(
        "recovery_status"
    )

    lines = [
        "❌ Workflow 未能成功完成。",
    ]

    if failed_tool:
        lines.append(
            f"失败工具：{failed_tool}"
        )

    if recovery_status:
        lines.append(
            f"Recovery 状态：{recovery_status}"
        )

    if error:
        lines.append(
            f"错误原因：{error}"
        )

    return "\n".join(lines)


def _extract_latest_ai_message(
    messages: list[BaseMessage],
) -> str:

    for message in reversed(messages):

        if not isinstance(
            message,
            AIMessage,
        ):
            continue

        content = message.content

        if isinstance(
            content,
            str,
        ):

            text = content.strip()

            if text:
                return text

    return ""