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


def create_ticket_agent(
    *,
    model: ChatOpenAI,
    tools: list[BaseTool],
    middleware: list[Any] | None = None,
):
    """
    创建 Ticket Specialist Agent。

    职责：
        - 查询工单
        - 创建工单
        - 更新工单
        - 工单生命周期处理

    当前工具：
        - get_ticket
        - create_ticket
        - update_ticket

    Tool Calling：
        由内部 LLM 自主决定。

    Risk / HITL：
        不由 Ticket Agent 自己判断，
        继续交给统一 Tool Governance。
    """

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=middleware or [],
        context_schema=EnterpriseAgentContext,
    )

    async def ticket_agent_node(
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
            "current_agent": "ticket_agent",
            "task_status": "completed",
        }

    return ticket_agent_node


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
        "Use memories as contextual information only. "
        "Before creating or updating a ticket, verify "
        "the current ticket state using the appropriate "
        "ticketing tool."
    )