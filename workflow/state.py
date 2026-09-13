from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


@dataclass(frozen=True, slots=True)
class EnterpriseAgentContext:
    """
    Enterprise-Agent 的运行时上下文。

    Context 表示当前 Workflow / Agent Run 的可信运行时信息，
    不属于 Graph State，也不属于 Long-term Memory 内容。

    user_id:
        当前请求所属用户。
        用于：
            - Long-term Memory Scope
            - Logging / Audit
            - 用户级数据隔离

    user_role:
        当前用户角色。
        用于：
            - PermissionPolicy
            - Tool Exposure

    tenant_id:
        当前租户。
        后续用于：
            - Multi-Tenant Isolation
            - Tool / Data Scope
            - Memory Scope
    """

    user_id: str
    user_role: str
    tenant_id: str | None = None


class EnterpriseAgentState(TypedDict, total=False):
    """
    Enterprise-Agent 的 LangGraph 全局 State。

    State 表示：
        当前 Workflow 在运行过程中不断变化、
        并需要在不同 Node / Agent 之间共享的数据。

    注意：
        user_id
        user_role
        tenant_id

    不放在 State 中，而由 EnterpriseAgentContext 提供。

    State 用于支撑：

        - Conversation
        - Long-term Memory
        - Supervisor
        - Specialist Agents
        - Handoff
        - Tool Execution
        - HITL
        - Error Handling
        - Workflow Recovery
        - Re-route
        - Parallel Execution
        - Checkpoint / Resume

    设计原则：

        Middleware 内部的 Tool / Model Retry Attempt
        不直接作为 Graph State。

        RetryMiddleware / ToolErrorMiddleware
        负责执行级 Retry 与错误处理；

        EnterpriseAgentState
        负责跨 Node / Agent / Checkpoint
        需要共享和持久化的 Workflow 状态。
    """

    # ==============================================================
    # Conversation State
    # ==============================================================

    messages: list[BaseMessage]

    # ==============================================================
    # Long-term Memory
    # ==============================================================

    # Memory Retrieval Node 在 Workflow 开始阶段写入。
    retrieved_memories: list[str]

    # Memory Persistence Node 写入。
    #
    # operation:
    #   add
    #   update
    #   ignore
    #   error
    #
    memory_persisted: bool
    memory_operation: str | None
    memory_persist_reason: str | None

    # ==============================================================
    # Agent / Workflow State
    # ==============================================================

    # 当前正在执行的 Specialist Agent。
    current_agent: str | None

    # Supervisor / Recovery 提供的下一步 Agent。
    next_agent: str | None

    # 当前 Workflow 状态。
    #
    # 推荐值：
    #   running
    #   completed
    #   failed
    #   interrupted
    task_status: str

    # ==============================================================
    # Routing / Handoff
    # ==============================================================

    # Supervisor / Handoff 的原因。
    handoff_reason: str | None

    # ==============================================================
    # Tool / Execution State
    # ==============================================================

    # 用于跨 Node 聚合 Tool 结果。
    #
    # Tool 本身的 retry attempt
    # 不在这里维护。
    tool_results: list[Any]

    # ==============================================================
    # Human-in-the-Loop
    # ==============================================================

    # 当前 Workflow 是否需要人工审批。
    approval_required: bool

    # 当前审批状态。
    #
    # 推荐值：
    #   pending
    #   approved
    #   rejected
    #   edited
    approval_status: str | None

    # ==============================================================
    # Error / Recovery State
    # ==============================================================

    # 当前 Workflow 最近一次结构化错误。
    #
    # ToolErrorMiddleware / Agent Runtime
    # 可以将最终失败转换为该字段。
    error: str | None

    # --------------------------------------------------------------
    # Failure Context
    # --------------------------------------------------------------

    # 最近一次发生错误的 Graph Node。
    last_failed_node: str | None

    # 最近一次失败的 Tool。
    #
    # 如果失败发生在 Agent Node 本身而非 Tool，
    # 则可以为 None。
    last_failed_tool: str | None

    # --------------------------------------------------------------
    # Workflow-level Recovery
    # --------------------------------------------------------------

    # 整个 Workflow 已经执行了多少次 Recovery。
    #
    # 注意：
    #
    #   recovery_attempts
    #       != Tool Retry Count
    #
    # Tool / Model Retry 由 Middleware 内部负责，
    # 不与 Workflow Recovery 共用计数。
    recovery_attempts: int

    # Recovery Node 决定采取的恢复策略。
    #
    # 当前支持：
    #   retry
    #   reroute
    #   human_review
    #   failed
    #   recovered
    recovery_status: str | None

    # Recovery 决策原因。
    #
    # 用于：
    #   - Logging
    #   - Audit
    #   - Debugging
    #   - Recovery Analysis
    recovery_reason: str | None

    # 是否需要通过 HITL / Checkpoint Resume
    # 恢复当前 Workflow。
    resume_required: bool


    # ==============================================================
    # Finalization
    # ==============================================================

    """
    Finalizer 生成的最终用户答案。
    """

    final_answer: str | None

    """
    最终执行事实摘要。

    供：
        - Finalizer
        - API
        - Logging
        - Audit
        - UI
    """

    execution_summary: dict[str, Any]