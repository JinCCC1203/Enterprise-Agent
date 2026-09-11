from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from memories.long_memory.manager import MemoryManager

from workflow.nodes.knowledge_agent import (
    create_knowledge_agent,
)
from workflow.nodes.memory import (
    create_memory_retrieval_node,
)
from workflow.nodes.operations_agent import (
    create_operations_agent,
)
from workflow.nodes.research_agent import (
    create_research_agent,
)
from workflow.nodes.supervisor import (
    create_supervisor_node,
)
from workflow.nodes.ticket_agent import (
    create_ticket_agent,
)
from workflow.routing.supervisor_router import (
    supervisor_router,
)
from workflow.state import (
    EnterpriseAgentContext,
    EnterpriseAgentState,
)


def build_enterprise_graph(
    *,
    model: ChatOpenAI,
    knowledge_tools: list[BaseTool],
    operations_tools: list[BaseTool],
    ticket_tools: list[BaseTool],
    research_tools: list[BaseTool],
    memory_manager: MemoryManager | None = None,
    middleware: list[Any] | None = None,
):
    """
    构建 Enterprise Multi-Agent Workflow。

    当前流程：

        START
          ↓
        Memory Retrieval
          ↓
        Supervisor
          ↓
        Conditional Routing
          ├── Knowledge Agent
          ├── Operations Agent
          ├── Ticket Agent
          ├── Research Agent
          └── END

    Specialist Agent 内部：

        Specialist Agent
              ↓
        LLM Tool Calling
              ↓
        Tool Governance
              ↓
        Tool Execution

    说明：
        1. Supervisor 负责“选择哪个 Specialist”。
        2. Specialist 负责“如何完成任务”。
        3. Tool Calling 由 Specialist 内部 LLM 完成。
        4. Router 只负责 Conditional Routing。
        5. Memory Retrieval 是 Workflow 级别操作。
        6. user_id / role / tenant_id 等安全上下文来自
           EnterpriseAgentContext，而不是 Graph State。
    """

    graph = StateGraph(
        EnterpriseAgentState,
        context_schema=EnterpriseAgentContext,
    )

    # ==============================================================
    # Middleware
    # ==============================================================

    shared_middleware = middleware or []

    # ==============================================================
    # 1. Memory Retrieval Node
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

    # ==============================================================
    # 2. Supervisor Node
    # ==============================================================

    supervisor_node = create_supervisor_node(
        model=model,
    )

    graph.add_node(
        "supervisor",
        supervisor_node,
    )

    # ==============================================================
    # 3. Knowledge Specialist Agent
    # ==============================================================

    knowledge_agent_node = create_knowledge_agent(
        model=model,
        tools=knowledge_tools,
        middleware=shared_middleware,
    )

    graph.add_node(
        "knowledge_agent",
        knowledge_agent_node,
    )

    # ==============================================================
    # 4. Operations Specialist Agent
    # ==============================================================

    operations_agent_node = create_operations_agent(
        model=model,
        tools=operations_tools,
        middleware=shared_middleware,
    )

    graph.add_node(
        "operations_agent",
        operations_agent_node,
    )

    # ==============================================================
    # 5. Ticket Specialist Agent
    # ==============================================================

    ticket_agent_node = create_ticket_agent(
        model=model,
        tools=ticket_tools,
        middleware=shared_middleware,
    )

    graph.add_node(
        "ticket_agent",
        ticket_agent_node,
    )

    # ==============================================================
    # 6. Research Specialist Agent
    # ==============================================================

    research_agent_node = create_research_agent(
        model=model,
        tools=research_tools,
        middleware=shared_middleware,
    )

    graph.add_node(
        "research_agent",
        research_agent_node,
    )

    # ==============================================================
    # 7. START
    # ==============================================================

    if memory_manager is not None:

        graph.add_edge(
            START,
            "memory_retrieval",
        )

        graph.add_edge(
            "memory_retrieval",
            "supervisor",
        )

    else:

        # Memory 系统未启用时，
        # 允许直接进入 Supervisor。
        graph.add_edge(
            START,
            "supervisor",
        )

    # ==============================================================
    # 8. Supervisor → Conditional Routing
    # ==============================================================

    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "knowledge_agent": "knowledge_agent",
            "operations_agent": "operations_agent",
            "ticket_agent": "ticket_agent",
            "research_agent": "research_agent",
            "end": END,
        },
    )

    # ==============================================================
    # 9. Specialist → END
    #
    # 当前阶段先让 Specialist 完成后结束。
    #
    # 后续加入：
    #   - recovery
    #   - retry
    #   - handoff
    #   - parallel execution
    #   - memory persist
    #
    # 后，这里会进一步扩展。
    # ==============================================================

    graph.add_edge(
        "knowledge_agent",
        END,
    )

    graph.add_edge(
        "operations_agent",
        END,
    )

    graph.add_edge(
        "ticket_agent",
        END,
    )

    graph.add_edge(
        "research_agent",
        END,
    )

    return graph.compile()