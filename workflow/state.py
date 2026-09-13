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
        - HITL
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

    tool_results: list[Any]

    # ==============================================================
    # Human-in-the-Loop
    # ==============================================================

    approval_required: bool
    approval_status: str | None

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
    resume_required: bool

    # ==============================================================
    # Finalization
    # ==============================================================

    final_answer: str | None

    """
    Runtime Execution Truth。

    Finalizer 生成并写入。
    """

    execution_summary: dict[str, Any]