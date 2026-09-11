from __future__ import annotations

from typing import Any
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
from langchain.agents import create_agent

from workflow.state import (
    EnterpriseAgentContext,
    EnterpriseAgentState,
)


def create_research_agent(
    *,
    model: ChatOpenAI,
    tools: list[BaseTool],
    middleware: list[Any] | None = None,
):
    """
    创建 Research / Web Specialist Agent。

    职责：
        - 外部信息检索
        - 当前实时信息查询
        - 公开技术资料检索
        - 官方文档检索
        - 新闻 / 外部知识研究

    当前预期工具：
        - web_search

    注意：
        Web Search 是 Tool。
        Research Agent 是业务领域 Agent。

        两者不能混淆。
    """

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=middleware or [],
        context_schema=EnterpriseAgentContext,
    )

    async def research_agent_node(
        state: EnterpriseAgentState,
        runtime: Runtime[EnterpriseAgentContext],
        config: Any,
    ) -> dict[str, Any]:

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

        agent_messages = list(messages)

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

        return {
            "messages": result_messages,
            "current_agent": "research_agent",
            "task_status": "completed",
        }

    return research_agent_node


def _build_memory_context(
    memories: list[str],
) -> str:
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
        "Treat these memories only as contextual hints. "
        "For current or time-sensitive information, "
        "always verify facts using external search results."
    )