from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from memories.long_memory.manager import MemoryManager
from policies.permission import PermissionPolicy
from tools_manager.registry import ToolRegistry

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
        registry: ToolRegistry,
        permission_policy: PermissionPolicy,
        middleware: list[Any] | None = None,
        memory_manager: MemoryManager | None = None,
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

        Global ToolRegistry
              ↓
        ToolRegistryView
              ↓
        Specialist Tool Scope
              ↓
        PermissionPolicy
              ↓
        DynamicToolMiddleware
              ↓
        LLM Tool Calling
              ↓
        RiskPolicy / HITL
              ↓
        Tool Execution
    """

    graph = StateGraph(
        EnterpriseAgentState,
        context_schema=EnterpriseAgentContext,
    )

    shared_middleware = list(
        middleware or []
    )

    # ==============================================================
    # 1. Memory Retrieval
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
    # 2. Supervisor
    # ==============================================================

    supervisor_node = create_supervisor_node(
        model=model,
    )

    graph.add_node(
        "supervisor",
        supervisor_node,
    )

    # ==============================================================
    # 3. Knowledge Agent
    # ==============================================================

    knowledge_agent_node = create_knowledge_agent(
        model=model,
        registry=registry,
        permission_policy=permission_policy,
        middleware=shared_middleware,
    )

    graph.add_node(
        "knowledge_agent",
        knowledge_agent_node,
    )

    # ==============================================================
    # 4. Operations Agent
    # ==============================================================

    operations_agent_node = create_operations_agent(
        model=model,
        registry=registry,
        permission_policy=permission_policy,
        middleware=shared_middleware,
    )

    graph.add_node(
        "operations_agent",
        operations_agent_node,
    )

    # ==============================================================
    # 5. Ticket Agent
    # ==============================================================

    ticket_agent_node = create_ticket_agent(
        model=model,
        registry=registry,
        permission_policy=permission_policy,
        middleware=shared_middleware,
    )

    graph.add_node(
        "ticket_agent",
        ticket_agent_node,
    )

    # ==============================================================
    # 6. Research Agent
    # ==============================================================

    research_agent_node = create_research_agent(
        model=model,
        registry=registry,
        permission_policy=permission_policy,
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
    # 当前阶段：
    # Specialist 完成后结束 Workflow。
    #
    # 后续再增加：
    #   - Recovery
    #   - Re-route
    #   - Handoff
    #   - Memory Persist
    #   - Parallel Execution
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