from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from memories.long_memory.manager import MemoryManager
from workflow.nodes.agent import agent_node
from workflow.nodes.memory import create_memory_retrieval_node
from workflow.state import (
    EnterpriseAgentContext,
    EnterpriseAgentState,
)


def build_single_agent_graph(
    *,
    agent: Any,
    memory_manager: MemoryManager | None = None,
):
    """
    构建当前过渡阶段的 Single-Agent Graph。

    当前流程：

        START
          ↓
        Memory Retrieval
          ↓
        Agent
          ↓
        END

    memory_manager 为 None 时，
    可以暂时跳过 Memory Retrieval，
    方便在数据库 / pgvector 尚未启动时调试 Agent Runtime。
    """

    graph = StateGraph(
        EnterpriseAgentState,
        context_schema=EnterpriseAgentContext,
    )

    # ==============================================================
    # Memory Retrieval
    # ==============================================================

    if memory_manager is not None:
        memory_retrieval_node = (
            create_memory_retrieval_node(
                memory_manager=memory_manager,
                top_k=5,
            )
        )

        graph.add_node(
            "memory_retrieval",
            memory_retrieval_node,
        )

        graph.add_edge(
            START,
            "memory_retrieval",
        )

        first_node = "memory_retrieval"

    else:
        first_node = "agent"

        graph.add_edge(
            START,
            "agent",
        )

    # ==============================================================
    # Agent
    # ==============================================================

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

    if first_node == "memory_retrieval":
        graph.add_edge(
            "memory_retrieval",
            "agent",
        )

    graph.add_edge(
        "agent",
        END,
    )

    return graph.compile()

