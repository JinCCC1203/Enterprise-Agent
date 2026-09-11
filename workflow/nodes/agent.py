from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig

from workflow.state import (
    EnterpriseAgentContext,
    EnterpriseAgentState,
)


async def agent_node(
    state: EnterpriseAgentState,
    runtime: Runtime[EnterpriseAgentContext],
    config: RunnableConfig,
    *,
    agent: Any,
) -> dict[str, Any]:
    """
    LangGraph Agent Node。

    当前阶段：
        直接复用已有 LangChain create_agent()。

    Runtime Context：
        LangGraph Runtime Context
            ↓
        agent.ainvoke(..., context=...)
            ↓
        create_agent Middleware
            ↓
        Permission / Memory / PII / HITL / Retry

    后续阶段：
        可以替换或扩展为：
            Supervisor
            Knowledge Agent
            Operations Agent
            Ticket Agent
            ...
    """

    messages: list[BaseMessage] = state.get(
        "messages",
        [],
    )

    # 将 LangGraph Runtime Context 继续传递给内部 LangChain Agent

    result = await agent.ainvoke(
        {
            "messages": messages,
        },
        config=config,
        context=runtime.context,
    )

    result_messages = result.get(
        "messages",
        [],
    )

    # Node 只返回本次产生的 State Update，
    return {
        "messages": result_messages,
        "task_status": "completed",
        "current_agent": "single_agent",
    }