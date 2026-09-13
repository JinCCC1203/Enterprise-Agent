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

    典型信息：

        - user_id
        - user_role
        - tenant_id

    这些信息由 Runtime 注入，并由 Agent / Middleware / Tool
    Governance 共同使用。
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
        - Multi-Specialist Orchestration
        - Specialist Handoff
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

    """
    当前正在执行的 Specialist。

    例如：

        operations_agent
        ticket_agent
        research_agent
        knowledge_agent
    """

    current_agent: str | None

    """
    Supervisor 当前规划的下一 Specialist。

    例如：

        operations_agent
        ticket_agent
        research_agent
        knowledge_agent

    当 Supervisor 判断整个任务已经完成时：

        next_agent = None
    """

    next_agent: str | None

    """
    当前子任务执行状态。

    推荐值：

        running
        completed
        failed
    """

    task_status: str

    """
    整个 Workflow 是否已经完成。

    False：
        仍然可能需要继续执行其他 Specialist。

    True：
        Supervisor 判断当前用户目标已经全部满足，
        可以进入 Memory Persist / Finalizer。
    """

    workflow_complete: bool

    """
    已经成功完成过的 Specialist 列表。

    示例：

        [
            "operations_agent",
            "ticket_agent",
        ]

    用于：

        - Supervisor 判断哪些子任务已经完成
        - 避免重复执行已经完成的 Specialist
        - 支撑 Multi-Agent Workflow Planning
    """

    completed_agents: list[str]

    # ==============================================================
    # Routing / Handoff
    # ==============================================================

    """
    Supervisor 对当前路由决策的解释。

    例如：

        "payment-service is degraded, so a P1 incident
         should be created by ticket_agent."
    """

    handoff_reason: str | None

    handoff_task: str | None

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
            "error_type": None,
            "error_message": None,
        }

    多 Specialist 执行时，该字段保存整个 Workflow
    中已经完成的 Tool Execution Facts。
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

    """
    当前 Workflow / Specialist 的错误信息。

    成功恢复以后，Supervisor / Specialist 可以清理该字段。
    """

    error: str | None

    """
    最后一次失败发生的 Graph Node。

    例如：

        ticket_agent
        operations_agent
    """

    last_failed_node: str | None

    """
    最后一次失败的 Tool。

    例如：

        create_ticket
        web_search
        get_service_health
    """

    last_failed_tool: str | None

    # ==============================================================
    # Workflow Recovery
    # ==============================================================

    """
    当前 Recovery 已经尝试的次数。

    注意：

        Tool Retry Attempt 不放在这里。

    recovery_attempts 表示 Workflow-level Recovery，
    而不是 Middleware-level Tool Retry。
    """

    recovery_attempts: int

    """
    当前 Recovery 状态。

    典型值：

        None
        retry
        reroute
        human_review
        failed
    """

    recovery_status: str | None

    """
    Recovery 决策原因。

    例如：

        "The failure appears to originate from a Tool,
         MCP server, network, or external service."
    """

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
        - next_agent
        - workflow_complete
        - completed_agents
        - approval_status
        - approval_events
        - tool_results
        - recovery_status
        - recovery_attempts
        - error
        - last_failed_node
        - last_failed_tool

    生成结构化执行摘要。
    """

    execution_summary: dict[str, Any]