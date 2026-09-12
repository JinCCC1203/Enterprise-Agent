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

    web_search 是通过 MCP 接入的外部搜索工具：

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

    Research Agent 的职责：

        - 外部公开信息检索
        - 实时信息查询
        - 技术资料研究
        - 官方文档搜索
        - 新闻 / 外部知识研究

    注意：
        Research Agent 本身不实现 HTTP 搜索逻辑。
        搜索能力完全由 MCP web_search Tool 提供。
    """

    # ------------------------------------------------------------------
    # 1. Research Registry View
    # ------------------------------------------------------------------

    registry_view = registry.create_view(
        RESEARCH_TOOL_SCOPE
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
    #
    # Registry 中的 web_search 必须已经被 MCP Client
    # list_tools() 发现并注册。
    # ------------------------------------------------------------------

    tools: list[BaseTool] = (
        registry_view.get_all_tools()
    )

    # ------------------------------------------------------------------
    # 6. 明确验证 web_search 已注册
    #
    # 因为 Research Agent 的核心依赖就是 web_search。
    # ------------------------------------------------------------------

    if not registry_view.contains(
        "web_search"
    ):
        raise RuntimeError(
            "Research Agent requires the "
            "'web_search' MCP tool, but it is not "
            "available in the ToolRegistry."
        )

    # ------------------------------------------------------------------
    # 7. LangChain Agent
    # ------------------------------------------------------------------

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=agent_middleware,
        context_schema=EnterpriseAgentContext,
    )

    # ------------------------------------------------------------------
    # 8. LangGraph Node
    # ------------------------------------------------------------------

    async def research_agent_node(
        state: EnterpriseAgentState,
        runtime: Runtime[EnterpriseAgentContext],
        config: Any,
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

        return {
            "messages": result.get(
                "messages",
                [],
            ),
            "current_agent": "research_agent",
            "task_status": "completed",
        }

    return research_agent_node


def _build_memory_context(
    memories: list[str],
) -> str:
    """
    构造长期记忆上下文。

    对 Research Agent：

        Memory
          = 历史上下文

        web_search
          = 当前外部事实的主要来源
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