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


KNOWLEDGE_TOOL_SCOPE = frozenset(
    {
        "rag_search",
    }
)


def create_knowledge_agent(
    *,
    model: ChatOpenAI,
    registry: ToolRegistry,
    permission_policy: PermissionPolicy,
    middleware: list[Any] | None = None,
):
    """
    创建 Knowledge Specialist Agent。

    Specialist Scope:
        rag_search

    职责：
        - 企业知识库查询
        - 内部文档检索
        - RAG 问答
        - 企业知识分析

    Tool Governance：

        Global ToolRegistry
                ↓
        Knowledge ToolRegistryView
                ↓
            rag_search
                ↓
        PermissionPolicy
                ↓
        DynamicToolMiddleware
                ↓
        LLM Tool Calling
    """

    # ------------------------------------------------------------------
    # 1. 创建 Knowledge Specialist 的受限 Registry View
    # ------------------------------------------------------------------

    registry_view = registry.create_view(
        KNOWLEDGE_TOOL_SCOPE
    )

    # ------------------------------------------------------------------
    # 2. 创建 Tool Exposure
    # ------------------------------------------------------------------

    tool_exposure = PermissionBasedToolExposure(
        registry_view=registry_view,
        permission_policy=permission_policy,
    )

    # ------------------------------------------------------------------
    # 3. 创建 Dynamic Tool Middleware
    # ------------------------------------------------------------------

    dynamic_tools = DynamicToolMiddleware(
        tool_exposure=tool_exposure,
    )

    # ------------------------------------------------------------------
    # 4. 组装 Middleware
    #
    # 注意：
    # middleware 参数中不要再次放入旧的
    # DynamicToolMiddleware。
    # ------------------------------------------------------------------

    agent_middleware = list(
        middleware or []
    )

    agent_middleware.append(
        dynamic_tools
    )

    # ------------------------------------------------------------------
    # 5. Specialist Agent 初始 Tool 集合
    #
    # 这里使用 Registry View，而不是 Global Registry。
    # ------------------------------------------------------------------

    tools: list[BaseTool] = (
        registry_view.get_all_tools()
    )

    # ------------------------------------------------------------------
    # 6. 创建 LangChain Agent
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

    async def knowledge_agent_node(
        state: EnterpriseAgentState,
        runtime: Runtime[EnterpriseAgentContext],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """
        Knowledge Specialist Graph Node。
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
            "current_agent": "knowledge_agent",
            "task_status": "completed",
        }

    return knowledge_agent_node


def _build_memory_context(
    memories: list[str],
) -> str:
    """
    构造长期记忆上下文。

    Long-term Memory 只是辅助上下文，
    不能替代当前 RAG 检索结果。
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
        "For enterprise knowledge questions, use the "
        "current enterprise knowledge base through "
        "rag_search as the authoritative source."
    )