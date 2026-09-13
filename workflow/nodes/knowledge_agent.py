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
        ToolRegistryView
                ↓
        Specialist Scope
                ↓
        PermissionPolicy
                ↓
        DynamicToolMiddleware
                ↓
        LLM Tool Calling
                ↓
        RiskPolicy / HITL
                ↓
        Tool Retry
                ↓
        Tool Error
                ↓
        Workflow Recovery
    """

    # ==============================================================
    # 1. Specialist Registry View
    # ==============================================================

    registry_view = registry.create_view(
        KNOWLEDGE_TOOL_SCOPE
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
    #
    # Specialist-specific middleware，放在当前 Agent middleware 最前面。
    # ==============================================================

    dynamic_tools = DynamicToolMiddleware(
        tool_exposure=tool_exposure,
        agent_name="knowledge_agent",
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

        # ----------------------------------------------------------
        # Agent Runtime
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # 只检查本次 Agent Invocation 新产生的消息
        #
        # 防止 Recovery retry 时，
        # 上一轮失败留下的 ToolMessage 污染当前结果。
        # ----------------------------------------------------------

        new_messages = result_messages[
            len(agent_messages):
        ]

        failure = extract_tool_execution_error(
            new_messages
        )

        if failure is not None:
            return {
                "messages": result_messages,
                "current_agent": "knowledge_agent",
                "task_status": "failed",
                "error": failure["error"],
                "last_failed_node": "knowledge_agent",
                "last_failed_tool": (
                    failure["last_failed_tool"]
                ),
            }

        # ----------------------------------------------------------
        # Normal Completion
        # ----------------------------------------------------------

        return {
            "messages": result_messages,
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