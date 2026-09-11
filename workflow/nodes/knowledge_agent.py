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


def create_knowledge_agent(
    *,
    model: ChatOpenAI,
    tools: list[BaseTool],
    middleware: list[Any] | None = None,
):
    """
    创建 Knowledge Specialist Agent。

    职责：
        - 企业知识库问答
        - 企业文档检索
        - RAG 查询
        - 基于内部知识进行回答

    工具：
        - rag_search

    注意：
        Tool Calling 由该 Agent 内部的 LLM 决定。
        本 Node 不自己判断何时调用 rag_search。
    """

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=middleware or [],
        context_schema=EnterpriseAgentContext,
    )

    async def knowledge_agent_node(
        state: EnterpriseAgentState,
        runtime: Runtime[EnterpriseAgentContext],
        config: Any,
    ) -> dict[str, Any]:
        """
        Knowledge Agent Node。

        Graph State
            ↓
        Agent
            ↓
        Tool Calling
            ↓
        Result
            ↓
        Graph State
        """

        messages = state.get(
            "messages",
            [],
        )

        retrieved_memories = state.get(
            "retrieved_memories",
            [],
        )

        # 将 Supervisor 之前检索到的长期记忆
        # 作为当前 Specialist 的上下文之一。
        memory_context = _build_memory_context(
            retrieved_memories
        )

        # Knowledge Agent 的输入使用当前 workflow
        # message history + memory context。
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
            "current_agent": "knowledge_agent",
            "task_status": "completed",
        }

    return knowledge_agent_node


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
        "Use these memories only as contextual hints. "
        "Do not treat them as authoritative facts when "
        "they conflict with retrieved enterprise knowledge."
    )