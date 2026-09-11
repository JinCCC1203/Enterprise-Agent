from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from workflow.nodes.agent import agent_node
from workflow.state import (
    EnterpriseAgentContext,
    EnterpriseAgentState,
)


def build_single_agent_graph(
    *,
    agent: Any,
):
    """
    构建第一阶段 LangGraph。
    当前：

        START
          ↓
        Agent
          ↓
        END
    Runtime Context：

        user_id
        user_role
        tenant_id

    不进入 Graph State，而通过 LangGraph Runtime Context 注入。

    后续扩展：

        START
          ↓
        Memory Retrieval
          ↓
        Supervisor
          ↓
        Specialist
          ↓
        Memory Persistence
          ↓
        END
    """

    graph = StateGraph(
        EnterpriseAgentState,
        context_schema=EnterpriseAgentContext,
    )

    # Agent Node
    # 使用闭包注入当前 create_agent() 实例。
    async def run_agent(
        state: EnterpriseAgentState,
        runtime,
        config,
    ) -> dict[str, Any]:

        return await agent_node(
            state,
            runtime,
            config,
            agent=agent,
        )

    graph.add_node(
        "agent",
        run_agent,
    )

    graph.add_edge(
        START,
        "agent",
    )

    graph.add_edge(
        "agent",
        END,
    )

    return graph.compile()