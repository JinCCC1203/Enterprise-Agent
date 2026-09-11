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

    State 后续用于支撑：
        - Long-term Memory
        - Supervisor
        - Specialist Agents
        - Handoff
        - Tool Execution
        - HITL
        - Retry / Recovery
        - Parallel Execution
        - Checkpoint / Resume
    """

    # ==============================================================
    # Conversation State
    # ==============================================================

    messages: list[BaseMessage]

    # ==============================================================
    # Long-term Memory
    #
    # Memory Retrieve Node 在 Workflow 开始阶段写入。
    # 后续整个 Workflow 共享。
    #
    # 当前使用字符串表示，后续可以根据实际需要扩展为
    # 更结构化的 Memory 对象。
    # ==============================================================

    retrieved_memories: list[str]

    # ==============================================================
    # Agent / Workflow State
    # ==============================================================

    current_agent: str | None

    next_agent: str | None

    task_status: str

    # ==============================================================
    # Routing / Handoff
    #
    # Supervisor 决定下一步 Agent 时使用。
    # ==============================================================

    handoff_reason: str | None

    # ==============================================================
    # Tool / Execution State
    #
    # 后续用于：
    # - Tool Result Aggregation
    # - Multi-step Execution
    # - Parallel Execution
    # - Specialist Collaboration
    # ==============================================================

    tool_results: list[Any]

    # ==============================================================
    # Human-in-the-Loop
    #
    # 后续用于：
    # - RiskPolicy
    # - HITL
    # - Approval / Reject / Edit
    # ==============================================================

    approval_required: bool

    approval_status: str | None

    # ==============================================================
    # Error / Recovery State
    #
    # 后续用于：
    # - Tool Error
    # - Retry
    # - Recovery
    # - Re-route
    # - Human Escalation
    # ==============================================================

    error: str | None

    retry_count: int

    # ==============================================================
    # Final Output
    # ==============================================================

    final_answer: str | None