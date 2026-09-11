from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


@dataclass(frozen=True, slots=True)
class EnterpriseAgentContext:
    """
    LangGraph / LangChain Agent 的运行时上下文。

    这些信息属于当前请求的可信 Runtime Context，
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
        后续多租户场景使用。
    """

    user_id: str
    user_role: str
    tenant_id: str | None = None


class EnterpriseAgentState(TypedDict, total=False):
    """
    Enterprise-Agent 的 LangGraph 全局 State。

    State 表示：
        当前 Workflow 在运行过程中不断变化的数据。

    注意：
        user_id / user_role / tenant_id
        不放在这里，而由 EnterpriseAgentContext 提供。
    """

    # Conversation State
    messages: list[BaseMessage]

    # Long-term Memory
    # 当前 Workflow 开始阶段由 Memory Retrieve Node 写入。
    # 后续整个 Workflow 共享。
    retrieved_memories: list[str]

    # Agent / Workflow State
    current_agent: str | None

    task_status: str

    # Tool / Execution State
    # 后续用于：
    # - Tool Result Aggregation
    # - Multi-step Execution
    # - Parallel Execution
    # - Recovery
    tool_results: list[Any]

    # Error / Recovery State
    error: str | None

    retry_count: int

    # Final Output
    final_answer: str | None