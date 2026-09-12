from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from middlewares.dynamic_tools import DynamicToolMiddleware
from policies.permission import PermissionPolicy
from tools_manager.registry import ToolRegistry
from tools_manager.tool_exposure import (
    PermissionBasedToolExposure,
)
from workflow.state import (
    EnterpriseAgentContext,
    EnterpriseAgentState,
)
from langchain_core.runnables import RunnableConfig


TICKET_TOOL_SCOPE = frozenset(
    {
        "get_ticket",
        "create_ticket",
        "update_ticket",
    }
)


def create_ticket_agent(
    *,
    model: ChatOpenAI,
    registry: ToolRegistry,
    permission_policy: PermissionPolicy,
    middleware: list[Any] | None = None,
):
    """
    创建 Ticket Specialist Agent。

    Specialist Scope:

        get_ticket
        create_ticket
        update_ticket

    职责：
        - 查询工单
        - 创建工单
        - 更新工单
        - 工单生命周期处理

    注意：

        Agent 不直接执行 RiskPolicy。

        具体 Tool Call：

            LLM
             ↓
            Tool Call
             ↓
            RiskPolicy
             ↓
            HITL / Execute
    """

    # ------------------------------------------------------------------
    # 1. Ticket Registry View
    # ------------------------------------------------------------------

    registry_view = registry.create_view(
        TICKET_TOOL_SCOPE
    )

    # ------------------------------------------------------------------
    # 2. Tool Exposure
    # ------------------------------------------------------------------

    tool_exposure = PermissionBasedToolExposure(
        registry_view=registry_view,
        permission_policy=permission_policy,
    )

    # ------------------------------------------------------------------
    # 3. Dynamic Tool Middleware
    # ------------------------------------------------------------------

    dynamic_tools = DynamicToolMiddleware(
        tool_exposure=tool_exposure,
        agent_name="ticket_agent",
    )

    # ------------------------------------------------------------------
    # 4. Middleware
    # ------------------------------------------------------------------

    agent_middleware = list(
        middleware or []
    )

    agent_middleware.append(
        dynamic_tools
    )

    # ------------------------------------------------------------------
    # 5. Scoped Tools
    # ------------------------------------------------------------------

    tools: list[BaseTool] = (
        registry_view.get_all_tools()
    )

    # ------------------------------------------------------------------
    # 6. LangChain Agent
    # ------------------------------------------------------------------

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=agent_middleware,
        context_schema=EnterpriseAgentContext,
    )

    # ------------------------------------------------------------------
    # 7. LangGraph Node
    # ------------------------------------------------------------------

    async def ticket_agent_node(
        state: EnterpriseAgentState,
        runtime: Runtime[EnterpriseAgentContext],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """
        Ticket Specialist Graph Node。
        """

        messages = state.get(
            "messages",
            [],
        )

        retrieved_memories = state.get(
            "retrieved_memories",
            [],
        )

        memory_context = _build_memory_context(
            retrieved_memories
        )

        agent_messages = list(
            messages
        )

        if memory_context:
            agent_messages.insert(
                0,
                HumanMessage(
                    content=memory_context
                ),
            )

        result = await agent.ainvoke(
            {
                "messages": agent_messages,
            },
            config=config,
            context=runtime.context,
        )

        return {
            "messages": result.get(
                "messages",
                [],
            ),
            "current_agent": "ticket_agent",
            "task_status": "completed",
        }

    return ticket_agent_node


def _build_memory_context(
    memories: list[str],
) -> str:
    """
    构造长期记忆上下文。

    Memory 只能作为历史上下文。
    对当前 Ticket 状态：
        get_ticket
    是权威来源。

    对 create/update：
        必须根据当前任务和工具结果决定，
        不能仅凭 Memory 自动执行写操作。
    """

    if not memories:
        return ""

    memory_text = "\n".join(
        f"- {memory}"
        for memory in memories
    )

    return (
        "Relevant long-term memories retrieved "
        "for this workflow:\n"
        f"{memory_text}\n\n"
        "Use these memories only as historical context. "
        "Before modifying an existing ticket, verify "
        "its current state using the appropriate ticketing "
        "tool. Do not treat long-term memory as authoritative "
        "for the current ticket state."
    )