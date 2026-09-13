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
        - 修改 Approval / Recovery 状态
    """

    async def finalizer_node(
        state: EnterpriseAgentState,
    ) -> dict[str, Any]:

        execution_summary = (
            build_execution_summary(
                state
            )
        )

        # ==========================================================
        # 1. Workflow / Runtime Facts
        # ==========================================================

        task_status = (
            execution_summary.get(
                "task_status"
            )
        )

        tool_results = (
            execution_summary.get(
                "tool_results",
                [],
            )
        )

        # ==========================================================
        # 2. Workflow Failure
        # ==========================================================

        if task_status == "failed":

            answer = _build_failure_answer(
                execution_summary
            )

            return {
                "execution_summary": (
                    execution_summary
                ),
                "final_answer": answer,
            }

        # ==========================================================
        # 3. Tool-based execution
        # ==========================================================

        if tool_results:

            answer = _build_tool_answer(
                state=state,
                execution_summary=(
                    execution_summary
                ),
            )

            return {
                "execution_summary": (
                    execution_summary
                ),
                "final_answer": answer,
            }

        # ==========================================================
        # 4. 普通 Agent 问答
        # ==========================================================

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
            "execution_summary": (
                execution_summary
            ),
            "final_answer": answer,
        }

    return finalizer_node


# ==============================================================
# Tool Execution Answer
# ==============================================================


def _build_tool_answer(
    *,
    state: EnterpriseAgentState,
    execution_summary: dict[str, Any],
) -> str:
    """
    构造 Tool 执行类任务的最终答案。

    核心原则：

        关键事实全部来自 Runtime State /
        Execution Facts，而不是由 LLM 猜测。
    """

    lines: list[str] = []

    # ==========================================================
    # Workflow Status
    # ==========================================================

    task_status = execution_summary.get(
        "task_status"
    )

    if task_status == "completed":

        lines.append(
            "✅ Workflow 已成功完成。"
        )

    else:

        lines.append(
            f"Workflow 状态：{task_status}"
        )

    # ==========================================================
    # Tool-level HITL
    #
    # 不再依赖 approval_required。
    #
    # approval_required 只表示当前是否还有 pending approval。
    #
    # 最终结果应该查看 approval_events。
    # ==========================================================

    _append_approval_facts(
        lines=lines,
        execution_summary=execution_summary,
    )

    # ==========================================================
    # Recovery
    # ==========================================================

    _append_recovery_facts(
        lines=lines,
        execution_summary=execution_summary,
    )

    # ==========================================================
    # Tool Results
    # ==========================================================

    _append_tool_results(
        lines=lines,
        execution_summary=execution_summary,
    )

    return "\n".join(lines)


# ==============================================================
# Approval Facts
# ==============================================================


def _append_approval_facts(
    *,
    lines: list[str],
    execution_summary: dict[str, Any],
) -> None:
    """
    将 Tool-level HITL 事实追加到最终答案。

    approval_events：
        整个 Workflow 的审批历史。

    approval_status：
        最近一次审批结果。

    approval_required：
        当前是否仍有 pending approval。

    Finalizer 在 Terminal 状态下主要依赖
    approval_events，而不是 approval_required。
    """

    approval_required = bool(
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

    approval_events = (
        execution_summary.get(
            "approval_events",
            [],
        )
    )

    # ----------------------------------------------------------
    # 当前仍在等待人工审批
    #
    # 正常 Terminal Finalizer 一般不会经过这里，
    # 但保留逻辑，增强鲁棒性。
    # ----------------------------------------------------------

    if approval_required:

        lines.append(
            "人工审批：当前仍在等待审批。"
        )

        if approval_status:

            lines.append(
                "当前审批状态："
                f"{approval_status}"
            )

        return

    # ----------------------------------------------------------
    # 没有审批事件
    # ----------------------------------------------------------

    if not approval_events:

        return

    # ----------------------------------------------------------
    # 最近一次审批事件
    # ----------------------------------------------------------

    latest_event = approval_events[-1]

    if not isinstance(
        latest_event,
        dict,
    ):
        return

    decision = latest_event.get(
        "decision"
    )

    tool_names = latest_event.get(
        "tool_names",
        [],
    )

    if not isinstance(
        tool_names,
        list,
    ):
        tool_names = []

    # ----------------------------------------------------------
    # Approved
    # ----------------------------------------------------------

    if decision == "approved":

        lines.append(
            "最近一次 Tool 审批：已通过。"
        )

    # ----------------------------------------------------------
    # Edited
    # ----------------------------------------------------------

    elif decision == "edited":

        lines.append(
            "最近一次 Tool 审批："
            "已通过修改后的请求。"
        )

    # ----------------------------------------------------------
    # Rejected
    # ----------------------------------------------------------

    elif decision == "rejected":

        lines.append(
            "最近一次 Tool 审批：已拒绝。"
        )

    # ----------------------------------------------------------
    # Unknown
    # ----------------------------------------------------------

    elif decision:

        lines.append(
            "最近一次 Tool 审批："
            f"{decision}"
        )

    # ----------------------------------------------------------
    # Tool Names
    # ----------------------------------------------------------

    if tool_names:

        normalized_names = [
            str(name)
            for name in tool_names
            if name
        ]

        if normalized_names:

            lines.append(
                "审批工具："
                + ", ".join(
                    normalized_names
                )
            )

    # ----------------------------------------------------------
    # Optional: current state consistency check
    #
    # approval_status 应与最新事件基本一致。
    # 如果不一致，不覆盖 Runtime Truth，
    # 只额外记录状态。
    # ----------------------------------------------------------

    if (
        approval_status
        and decision
        and approval_status != decision
    ):

        lines.append(
            "注意：当前审批状态与最近审批事件存在不一致，"
            "以审批事件历史为准。"
        )


# ==============================================================
# Recovery Facts
# ==============================================================


def _append_recovery_facts(
    *,
    lines: list[str],
    execution_summary: dict[str, Any],
) -> None:
    """
    将 Workflow Recovery 信息追加到最终答案。
    """

    recovery_status = (
        execution_summary.get(
            "recovery_status"
        )
    )

    recovery_attempts = (
        execution_summary.get(
            "recovery_attempts",
            0,
        )
    )

    recovery_reason = (
        execution_summary.get(
            "recovery_reason"
        )
    )

    if not recovery_status:

        return

    lines.append(
        "\nWorkflow Recovery："
        f"{recovery_status}"
    )

    if recovery_attempts:

        lines.append(
            f"Recovery Attempts："
            f"{recovery_attempts}"
        )

    if recovery_reason:

        lines.append(
            f"Recovery Reason："
            f"{recovery_reason}"
        )


# ==============================================================
# Tool Results
# ==============================================================


def _append_tool_results(
    *,
    lines: list[str],
    execution_summary: dict[str, Any],
) -> None:
    """
    将 Tool Execution Facts 追加到最终答案。
    """

    tool_results = (
        execution_summary.get(
            "tool_results",
            [],
        )
    )

    if not tool_results:

        return

    successful_tools: list[
        dict[str, Any]
    ] = []

    failed_tools: list[
        dict[str, Any]
    ] = []

    for result in tool_results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        if result.get(
            "error",
            False,
        ):

            failed_tools.append(
                result
            )

        else:

            successful_tools.append(
                result
            )

    # ----------------------------------------------------------
    # Successful Tools
    # ----------------------------------------------------------

    if successful_tools:

        lines.append(
            "\n已成功执行的工具："
        )

        for result in successful_tools:

            tool_name = result.get(
                "tool_name",
                "unknown",
            )

            content = result.get(
                "content"
            )

            lines.append(
                f"- {tool_name}"
            )

            lines.append(
                _format_tool_content(
                    content
                )
            )

    # ----------------------------------------------------------
    # Failed Tools
    # ----------------------------------------------------------

    if failed_tools:

        lines.append(
            "\n工具执行失败："
        )

        for result in failed_tools:

            tool_name = result.get(
                "tool_name",
                "unknown",
            )

            error_type = result.get(
                "error_type"
            )

            error_message = result.get(
                "error_message"
            )

            if error_type:

                lines.append(
                    f"- {tool_name} "
                    f"({error_type})"
                )

            else:

                lines.append(
                    f"- {tool_name}"
                )

            if error_message:

                lines.append(
                    f"  错误：{error_message}"
                )


# ==============================================================
# Failure Answer
# ==============================================================


def _build_failure_answer(
    execution_summary: dict[str, Any],
) -> str:
    """
    构造 Workflow Failure 最终答案。
    """

    lines: list[str] = [
        "❌ Workflow 未能成功完成。"
    ]

    # ----------------------------------------------------------
    # Failed Node
    # ----------------------------------------------------------

    failed_node = (
        execution_summary.get(
            "last_failed_node"
        )
    )

    if failed_node:

        lines.append(
            f"失败节点：{failed_node}"
        )

    # ----------------------------------------------------------
    # Failed Tool
    # ----------------------------------------------------------

    failed_tool = (
        execution_summary.get(
            "last_failed_tool"
        )
    )

    if failed_tool:

        lines.append(
            f"失败工具：{failed_tool}"
        )

    # ----------------------------------------------------------
    # Recovery
    # ----------------------------------------------------------

    _append_recovery_facts(
        lines=lines,
        execution_summary=execution_summary,
    )

    # ----------------------------------------------------------
    # Error
    # ----------------------------------------------------------

    error = execution_summary.get(
        "error"
    )

    if error:

        lines.append(
            f"错误原因：{error}"
        )

    # ----------------------------------------------------------
    # Approval
    #
    # 即使最终失败，也可以保留最近一次审批事实。
    # ----------------------------------------------------------

    _append_approval_facts(
        lines=lines,
        execution_summary=execution_summary,
    )

    return "\n".join(lines)


# ==============================================================
# Normal Agent Answer
# ==============================================================


def _extract_latest_ai_message(
    messages: list[BaseMessage],
) -> str:
    """
    普通 Agent 场景下，从消息历史中提取最后一个 AIMessage。

    注意：
        只有没有 Tool Execution Facts 的普通对话才走这里。
    """

    for message in reversed(
        messages
    ):

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


# ==============================================================
# Tool Content Formatting
# ==============================================================


def _format_tool_content(
    content: Any,
) -> str:
    """
    格式化 Tool 返回结果。

    不做 LLM 重写，避免丢失 Runtime Execution Truth。
    """

    if content is None:

        return "  执行结果：None"

    if isinstance(
        content,
        str,
    ):

        text = content.strip()

        if not text:

            return "  执行结果："

        return (
            "  执行结果："
            f"{text}"
        )

    return (
        "  执行结果："
        f"{content}"
    )