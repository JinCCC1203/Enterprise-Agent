from __future__ import annotations

import operator

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage


@dataclass(frozen=True, slots=True)
class EnterpriseAgentContext:
    """
    Enterprise-Agent 的运行时上下文。

    Context 表示当前 Workflow / Agent Run 的可信运行时信息，
    不属于 Graph State，也不属于 Long-term Memory 内容。
    """

    user_id: str
    user_role: str
    tenant_id: str | None = None


class EnterpriseAgentState(TypedDict, total=False):
    """
    Enterprise-Agent 的 LangGraph 全局 State。

    State 用于支撑：

        - Conversation
        - Long-term Memory
        - Supervisor
        - Specialist Agents
        - Handoff
        - Tool Execution
        - Tool-level HITL
        - Error Handling
        - Workflow Recovery
        - Re-route
        - Checkpoint / Resume
        - Finalization

    Middleware 内部的 Tool / Model Retry Attempt
    不直接作为 Graph State。
    """

    # ==============================================================
    # Conversation
    # ==============================================================

    messages: list[BaseMessage]

    # ==============================================================
    # Long-term Memory
    # ==============================================================

    retrieved_memories: list[str]

    memory_persisted: bool
    memory_operation: str | None
    memory_persist_reason: str | None

    # ==============================================================
    # Agent / Workflow
    # ==============================================================

    current_agent: str | None
    next_agent: str | None

    task_status: str

    # ==============================================================
    # Routing / Handoff
    # ==============================================================

    handoff_reason: str | None

    # ==============================================================
    # Tool / Execution
    # ==============================================================

    """
    Tool 执行事实。

    例如：

        {
            "tool_name": "create_ticket",
            "tool_call_id": "...",
            "content": "...",
            "error": False,
            ...
        }
    """

    tool_results: list[Any]

    # ==============================================================
    # Tool-level Human-in-the-Loop
    # ==============================================================

    """
    当前是否存在待人工处理的 Tool Approval。

    True：
        当前存在 pending approval。

    False：
        当前 Tool Approval 已经完成。
    """

    approval_required: bool

    """
    最近一次 Tool-level HITL 的决策结果。

    可能为：

        None
        approved
        edited
        rejected

    注意：
        该字段表示“最近一次决策”，
        不表示整个 Workflow 的永久审批状态。
    """

    approval_status: str | None

    """
    整个 Workflow 生命周期内的 Tool Approval Event History。

    使用 LangGraph reducer 追加事件，而不是覆盖旧事件。

    示例：

        [
            {
                "type": "tool_approval",
                "tool_names": ["create_ticket"],
                "decision": "approved",
            },
            {
                "type": "tool_approval",
                "tool_names": ["update_ticket"],
                "decision": "approved",
            },
            {
                "type": "tool_approval",
                "tool_names": ["send_notification"],
                "decision": "rejected",
            },
        ]
    """

    approval_events: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]

    # ==============================================================
    # Error / Failure
    # ==============================================================

    error: str | None

    last_failed_node: str | None
    last_failed_tool: str | None

    # ==============================================================
    # Workflow Recovery
    # ==============================================================

    recovery_attempts: int
    recovery_status: str | None
    recovery_reason: str | None

    """
    当前 Workflow 是否等待恢复后的继续执行。

    True：
        Workflow interrupted / waiting for resume。

    False：
        已经恢复或者不需要恢复。
    """

    resume_required: bool

    # ==============================================================
    # Finalization
    # ==============================================================

    """
    Finalizer 生成的最终用户可见答案。

    不再直接依赖 messages[-1]。
    """

    final_answer: str | None

    """
    Runtime Execution Truth。

    Finalizer 根据：

        - task_status
        - current_agent
        - approval_status
        - approval_events
        - tool_results
        - recovery_status
        - error

    生成结构化执行摘要。
    """

    execution_summary: dict[str, Any]