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

from workflow.utils.execution_facts import (
    collect_tool_results,
)


RESEARCH_TOOL_SCOPE = frozenset(
    {
        "web_search",
    }
)


def create_research_agent(
    *,
    model: ChatOpenAI,
    registry: ToolRegistry,
    permission_policy: PermissionPolicy,
    middleware: list[Any] | None = None,
):
    """
    创建 Research / Web Specialist Agent。

    Specialist Scope:
        web_search

    web_search 通过 MCP 接入：

        Research Agent
              ↓
          web_search
              ↓
        MCP Client
              ↓
        MCP Server
              ↓
        External Search Provider
              ↓
           Internet
    """

    # ==============================================================
    # 1. Specialist Registry View
    # ==============================================================

    registry_view = registry.create_view(
        RESEARCH_TOOL_SCOPE
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
        agent_name="research_agent",
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
    # 6. Validate web_search
    # ==============================================================

    if not registry_view.contains(
        "web_search"
    ):
        raise RuntimeError(
            "Research Agent requires the "
            "'web_search' MCP tool, but it is not "
            "available in the ToolRegistry."
        )

    # ==============================================================
    # 7. LangChain Agent
    # ==============================================================

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=agent_middleware,
        context_schema=EnterpriseAgentContext,
    )

    # ==============================================================
    # 8. LangGraph Node
    # ==============================================================

    async def research_agent_node(
        state: EnterpriseAgentState,
        runtime: Runtime[EnterpriseAgentContext],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """
        Research Specialist Graph Node。
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

        tool_results = collect_tool_results(
            result_messages
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
                "tool_results": tool_results,
            }

        return {
            "messages": result_messages,
            "current_agent": "ticket_agent",
            "task_status": "completed",
            "tool_results": tool_results,
        }

    return research_agent_node


def _build_memory_context(
    memories: list[str],
) -> str:
    """
    构造长期记忆上下文。

    Memory：
        历史上下文。

    web_search：
        当前外部、实时信息的主要来源。
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
        "Use these memories only as contextual hints. "
        "For current, external, or time-sensitive information, "
        "always verify the information through the web_search "
        "tool before presenting it as a current fact."
    )