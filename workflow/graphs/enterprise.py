from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from memories.long_memory.manager import MemoryManager
from policies.permission import PermissionPolicy
from tools_manager.registry import ToolRegistry

from workflow.nodes.human_review import (
    create_human_review_node,
)
from workflow.nodes.knowledge_agent import (
    create_knowledge_agent,
)
from workflow.nodes.memory import (
    create_memory_retrieval_node,
)
from workflow.nodes.memory_persist import (
    create_memory_persist_node,
)
from workflow.nodes.operations_agent import (
    create_operations_agent,
)
from workflow.nodes.recovery import (
    RecoveryPolicy,
    create_recovery_node,
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
from workflow.routing.human_review_router import (
    human_review_router,
)
from workflow.routing.recovery_router import (
    recovery_router,
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
    checkpointer: AsyncPostgresSaver | None = None,
):
    """
    构建 Enterprise Multi-Agent Workflow。

    ==============================================================
    主流程
    ==============================================================

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
          └── Research Agent

    ==============================================================
    Specialist 成功
    ==============================================================

        Specialist
            ↓
        Memory Persist
            ↓
           END

    ==============================================================
    Specialist 最终失败
    ==============================================================

        Specialist
            ↓
         Recovery
          ├── retry
          │     ↓
          │   原 Specialist
          │
          ├── reroute
          │     ↓
          │   Supervisor
          │     ↓
          │   LLM 重新规划
          │
          ├── human_review
          │     ↓
          │   Human Review
          │     ↓
          │   interrupt()
          │     ↓
          │   Checkpoint
          │     ↓
          │   Command(resume)
          │     ↓
          │   Human Review Router
          │     ├── retry
          │     ├── reroute
          │     └── reject
          │
          └── failed
                ↓
          Memory Persist
                ↓
               END

    ==============================================================
    Specialist Agent 内部
    ==============================================================

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
        RiskPolicy / Tool-level HITL
              ↓
        Tool Retry
              ↓
        Tool Error
              ↓
        Specialist Node
              ↓
        EnterpriseAgentState

    ==============================================================
    Persistence
    ==============================================================

        Graph State
              ↓
        PostgreSQL Checkpointer
              ↓
        Checkpoint / Interrupt / Resume / Recovery
    """

    # ==============================================================
    # 1. StateGraph
    # ==============================================================

    graph = StateGraph(
        EnterpriseAgentState,
        context_schema=EnterpriseAgentContext,
    )

    # ==============================================================
    # 2. Shared Middleware
    #
    # DynamicToolMiddleware 不放这里。
    #
    # 每个 Specialist 根据自己的 ToolRegistryView
    # 创建自己的 DynamicToolMiddleware。
    # ==============================================================

    shared_middleware = list(
        middleware or []
    )

    # ==============================================================
    # 3. Memory Retrieval
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
    # 4. Supervisor
    # ==============================================================

    supervisor_node = create_supervisor_node(
        model=model,
    )

    graph.add_node(
        "supervisor",
        supervisor_node,
    )

    # ==============================================================
    # 5. Knowledge Agent
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
    # 6. Operations Agent
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
    # 7. Ticket Agent
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
    # 8. Research Agent
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
    # 9. Memory Persistence
    # ==============================================================

    if memory_manager is not None:

        memory_persist_node = (
            create_memory_persist_node(
                memory_manager=memory_manager,
            )
        )

        graph.add_node(
            "memory_persist",
            memory_persist_node,
        )

    # ==============================================================
    # 10. Recovery
    # ==============================================================

    recovery_policy = RecoveryPolicy(
        max_recovery_attempts=2,
    )

    recovery_node = create_recovery_node(
        recovery_policy=recovery_policy,
    )

    graph.add_node(
        "recovery",
        recovery_node,
    )

    # ==============================================================
    # 11. Workflow-level Human Review
    #
    # 注意：
    #
    # 这不是 HumanInTheLoopMiddleware。
    #
    # Middleware：
    #     Tool-level HITL
    #
    # human_review：
    #     Workflow-level Recovery HITL
    # ==============================================================

    human_review_node = (
        create_human_review_node()
    )

    graph.add_node(
        "human_review",
        human_review_node,
    )

    # ==============================================================
    # 12. START
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
    # 13. Supervisor → Specialist
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
    # 14. Specialist Outcome Router
    #
    # completed:
    #     → Memory Persist / END
    #
    # failed:
    #     → Recovery
    # ==============================================================

    def specialist_outcome_router(
        state: EnterpriseAgentState,
    ) -> str:

        task_status = state.get(
            "task_status"
        )

        if task_status == "failed":
            return "recovery"

        if memory_manager is not None:
            return "memory_persist"

        return "end"

    specialist_routes = {
        "recovery": "recovery",
        "memory_persist": "memory_persist",
        "end": END,
    }

    # ==============================================================
    # 15. Specialist → Success / Recovery
    # ==============================================================

    for specialist_name in (
        "knowledge_agent",
        "operations_agent",
        "ticket_agent",
        "research_agent",
    ):
        graph.add_conditional_edges(
            specialist_name,
            specialist_outcome_router,
            specialist_routes,
        )

    # ==============================================================
    # 16. Recovery Router
    #
    # retry:
    #     → 原 Specialist
    #
    # reroute:
    #     → Supervisor
    #
    # human_review:
    #     → Human Review Node
    #
    # failed:
    #     → Memory Persist
    # ==============================================================

    recovery_routes: dict[str, str] = {
        "knowledge_agent": "knowledge_agent",
        "operations_agent": "operations_agent",
        "ticket_agent": "ticket_agent",
        "research_agent": "research_agent",
        "supervisor": "supervisor",
        "human_review": "human_review",
        "memory_persist": "memory_persist",
    }

    # 如果 memory_manager=None，
    # 当前测试模式没有 Memory Persist。
    if memory_manager is None:
        recovery_routes.pop(
            "memory_persist"
        )

    graph.add_conditional_edges(
        "recovery",
        recovery_router,
        recovery_routes,
    )

    # ==============================================================
    # 17. Human Review Router
    #
    # human_retry:
    #     → 原 Specialist
    #
    # human_reroute:
    #     → Supervisor
    #
    # human_rejected:
    #     → Memory Persist
    # ==============================================================

    human_review_routes: dict[str, str] = {
        "knowledge_agent": "knowledge_agent",
        "operations_agent": "operations_agent",
        "ticket_agent": "ticket_agent",
        "research_agent": "research_agent",
        "supervisor": "supervisor",
    }

    if memory_manager is not None:
        human_review_routes[
            "memory_persist"
        ] = "memory_persist"

    graph.add_conditional_edges(
        "human_review",
        human_review_router,
        human_review_routes,
    )

    # ==============================================================
    # 18. Memory Persist → END
    # ==============================================================

    if memory_manager is not None:

        graph.add_edge(
            "memory_persist",
            END,
        )

    # ==============================================================
    # 19. Compile
    # ==============================================================

    return graph.compile(
        checkpointer=checkpointer,
    )