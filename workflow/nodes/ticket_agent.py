from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
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
from workflow.utils.execution_errors import (
    extract_tool_execution_error,
)


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
    """

    # ==============================================================
    # 1. Specialist Registry View
    # ==============================================================

    registry_view = registry.create_view(
        TICKET_TOOL_SCOPE
    )

    # ==============================================================
    # 2. Tool Exposure
    # ==============================================================

    tool_exposure = PermissionBasedToolExposure(
        registry_view=registry_view,
        permission_policy=permission_policy,
    )

    # ==============================================================
    # 3. Dynamic Tool Middleware
    # ==============================================================

    dynamic_tools = DynamicToolMiddleware(
        tool_exposure=tool_exposure,
        agent_name="ticket_agent",
    )

    # ==============================================================
    # 4. Middleware Assembly
    # ==============================================================

    agent_middleware = [
        dynamic_tools,
        *(middleware or []),
    ]

    # ==============================================================
    # 5. Scoped Tools
    # ==============================================================

    tools: list[BaseTool] = (
        registry_view.get_all_tools()
    )

    # ==============================================================
    # 6. LangChain Agent
    # ==============================================================

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=agent_middleware,
        context_schema=EnterpriseAgentContext,
    )

    # ==============================================================
    # 7. LangGraph Node
    # ==============================================================

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

        result_messages = result.get(
            "messages",
            [],
        )

        # 只检查当前 invocation 新增的消息。
        new_messages = result_messages[
            len(agent_messages):
        ]

        failure = extract_tool_execution_error(
            new_messages
        )

        if failure is not None:
            return {
                "messages": result_messages,
                "current_agent": "ticket_agent",
                "task_status": "failed",
                "error": failure["error"],
                "last_failed_node": "ticket_agent",
                "last_failed_tool": (
                    failure["last_failed_tool"]
                ),
            }

        return {
            "messages": result_messages,
            "current_agent": "ticket_agent",
            "task_status": "completed",
        }

    return ticket_agent_node


def _build_memory_context(
    memories: list[str],
) -> str:
    """
    构造长期记忆上下文。

    Memory 只提供历史背景。

    当前 Ticket 状态：
        get_ticket
    是权威来源。

    对 create/update：
        必须依据当前任务以及 Tool Result，
        不能只根据历史 Memory 执行写操作。
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