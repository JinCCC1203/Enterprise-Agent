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

OPERATIONS_TOOL_SCOPE = frozenset(
    {
        "get_service_health",
    }
)


def create_operations_agent(
    *,
    model: ChatOpenAI,
    registry: ToolRegistry,
    permission_policy: PermissionPolicy,
    middleware: list[Any] | None = None,
):
    """
    创建 Operations Specialist Agent。

    Specialist Scope:
        get_service_health

    职责：
        - 服务健康检查
        - 当前服务状态
        - 基础运维诊断
        - 企业基础设施状态查询
    """

    # ------------------------------------------------------------------
    # 1. Operations Registry View
    # ------------------------------------------------------------------

    registry_view = registry.create_view(
        OPERATIONS_TOOL_SCOPE
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
        agent_name="operations_agent",
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

    async def operations_agent_node(
        state: EnterpriseAgentState,
        runtime: Runtime[EnterpriseAgentContext],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """
        Operations Specialist Graph Node。
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
            "current_agent": "operations_agent",
            "task_status": "completed",
        }

    return operations_agent_node


def _build_memory_context(
    memories: list[str],
) -> str:
    """
    构造长期记忆上下文。

    对运维状态而言，Memory 只能提供历史背景，
    当前 Tool 查询结果才是实时状态的权威来源。
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
        "For current service health or operational status, "
        "always use get_service_health and trust the "
        "current tool result over historical memory."
    )