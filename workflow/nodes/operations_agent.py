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


def create_operations_agent(
    *,
    model: ChatOpenAI,
    tools: list[BaseTool],
    middleware: list[Any] | None = None,
):
    """
    创建 Operations Specialist Agent。

    职责：
        - 服务健康检查
        - 系统状态查询
        - 运维诊断
        - 基础设施相关操作

    当前工具：
        - get_service_health

    未来可扩展：
        - get_service_metrics
        - get_deployment_status
        - get_incident
    """

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=middleware or [],
        context_schema=EnterpriseAgentContext,
    )

    async def operations_agent_node(
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
            "current_agent": "operations_agent",
            "task_status": "completed",
        }

    return operations_agent_node


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
        "Use these memories as contextual information only. "
        "Always rely on current operational tool results "
        "for real-time service status."
    )