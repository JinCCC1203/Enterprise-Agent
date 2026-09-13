from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from memories.long_memory.manager import MemoryManager
from policies.permission import PermissionPolicy
from tools_manager.registry import ToolRegistry

from workflow.nodes.finalizer import (
    create_finalizer_node,
)
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
from workflow.routing.specialist_router import (
    specialist_router,
)
from workflow.routing.supervisor_router import (
    supervisor_router,
)

from workflow.state import (
    EnterpriseAgentContext,
    EnterpriseAgentState,
)


# ==============================================================
# Specialist Names
# ==============================================================

SPECIALIST_NAMES = (
    "knowledge_agent",
    "operations_agent",
    "ticket_agent",
    "research_agent",
)


# ==============================================================
# Specialist Node Wrapper
# ==============================================================

def _wrap_specialist_node(
    specialist_name: str,
    node: Any,
):
    """
    包装 Specialist Node。

    目的：

        Specialist 成功执行后，
        将其记录到：

            state["completed_agents"]

    这样下一次 Supervisor 执行时，
    就能知道哪些 Specialist 已经完成。

    注意：

        - task_status == completed：
              才加入 completed_agents

        - task_status == failed：
              不加入 completed_agents

        - GraphInterrupt：
              直接向上抛出，不修改状态

    Specialist 本身仍然负责：
        - Tool Calling
        - Tool HITL
        - Tool Retry
        - Tool Error
        - Specialist-level Failure

    Wrapper 只负责 Workflow bookkeeping。
    """

    async def wrapped_node(
        state: EnterpriseAgentState,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:

        # ----------------------------------------------------------
        # 执行原始 Specialist Node
        # ----------------------------------------------------------

        result = await node(
            state,
            *args,
            **kwargs,
        )

        if not isinstance(
            result,
            dict,
        ):
            return result

        # ----------------------------------------------------------
        # Specialist 执行状态
        # ----------------------------------------------------------

        task_status = result.get(
            "task_status",
            state.get(
                "task_status",
                "running",
            ),
        )

        # ----------------------------------------------------------
        # Specialist 成功
        # ----------------------------------------------------------

        if task_status == "completed":

            completed_agents = list(
                state.get(
                    "completed_agents",
                    [],
                )
            )

            # 当前 Specialist 名称优先使用 wrapper
            # 的固定名称，而不是依赖 Specialist Node
            # 是否正确写入 current_agent。
            if (
                specialist_name
                not in completed_agents
            ):
                completed_agents.append(
                    specialist_name
                )

            result["completed_agents"] = (
                completed_agents
            )

        return result

    return wrapped_node


# ==============================================================
# Workflow Completion Router
# ==============================================================

def _workflow_completion_router(
    state: EnterpriseAgentState,
) -> str:
    """
    Workflow 完成后的最终路由。

    这里已经确认：

        workflow_complete == True

    因此只需要决定：

        memory_persist
            或
        finalizer
    """

    if state.get(
        "workflow_complete",
        False,
    ) is not True:
        # Defensive fallback。
        #
        # 正常情况下这里不会被调用，
        # 因为 specialist_router 只有在
        # workflow_complete=True 时才返回 completed。
        return "finalizer"

    memory_enabled = state.get(
        "_memory_enabled",
        False,
    )

    if memory_enabled:
        return "memory_persist"

    return "finalizer"


# ==============================================================
# Build Enterprise Graph
# ==============================================================

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
    Overall Workflow
    ==============================================================

        START
          ↓
        Memory Retrieval
          ↓
        Supervisor
          ↓
        Specialist
          ↓
        Specialist Outcome Router
          │
          ├── failed
          │      ↓
          │   Recovery
          │
          ├── completed
          │      ↓
          │   Supervisor
          │
          └── workflow_complete
                 ↓
          Memory Persist
                 ↓
             Finalizer
                 ↓
                END

    ==============================================================
    Multi-Specialist Flow
    ==============================================================

        Supervisor
             ↓
        operations_agent
             ↓
        Supervisor
             ↓
        ticket_agent
             ↓
        Supervisor
             ↓
        Finalizer

    例如：

        "查询 payment-service 当前健康状态，
         如果异常，请创建一个 P1 Incident 工单。"

    可能形成：

        Supervisor
            ↓
        operations_agent
            ↓
        get_service_health
            ↓
        Supervisor
            ↓
        ticket_agent
            ↓
        create_ticket
            ↓
        HITL
            ↓
        Supervisor
            ↓
        Finalizer

    ==============================================================
    Recovery
    ==============================================================

        Specialist
            ↓
         Recovery
          ├── retry
          │      ↓
          │   原 Specialist
          │
          ├── reroute
          │      ↓
          │   Supervisor
          │
          ├── human_review
          │      ↓
          │   Human Review
          │      ↓
          │   interrupt()
          │      ↓
          │   Command(resume)
          │
          └── failed
                 ↓
          Memory Persist
                 ↓
             Finalizer
                 ↓
                END

    ==============================================================
    Finalization
    ==============================================================

        Runtime State
             +
        Tool Execution Facts
             ↓
          Finalizer
             ↓
        final_answer
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
        _wrap_specialist_node(
            specialist_name="knowledge_agent",
            node=knowledge_agent_node,
        ),
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
        _wrap_specialist_node(
            specialist_name="operations_agent",
            node=operations_agent_node,
        ),
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
        _wrap_specialist_node(
            specialist_name="ticket_agent",
            node=ticket_agent_node,
        ),
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
        _wrap_specialist_node(
            specialist_name="research_agent",
            node=research_agent_node,
        ),
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
    # ==============================================================

    human_review_node = (
        create_human_review_node()
    )

    graph.add_node(
        "human_review",
        human_review_node,
    )

    # ==============================================================
    # 12. Finalizer
    # ==============================================================

    finalizer_node = (
        create_finalizer_node()
    )

    graph.add_node(
        "finalizer",
        finalizer_node,
    )

    # ==============================================================
    # 13. Specialist Completed Intermediate Node
    #
    # Specialist Router 返回：
    #
    #     completed
    #
    # 但 completed 之后还要根据：
    #
    #     workflow_complete
    #
    # 决定：
    #
    #     Memory Persist
    #     或
    #     Finalizer
    #
    # 所以这里增加一个很轻量的中间节点。
    # ==============================================================

    async def specialist_completed_node(
        state: EnterpriseAgentState,
    ) -> dict[str, Any]:
        """
        Specialist 已经成功，并且 Supervisor
        已经判断整个 Workflow 完成。

        该节点不修改业务 State，
        仅用于把 Graph 中的两个路由阶段分开。
        """

        return {}

    graph.add_node(
        "specialist_completed",
        specialist_completed_node,
    )

    # ==============================================================
    # 14. START
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
    # 15. Supervisor
    #
    # Supervisor 根据：
    #
    #     user query
    #     completed_agents
    #     tool_results
    #     current_agent
    #     workflow state
    #
    # 决定：
    #
    #     specialist
    #     end
    # ==============================================================

    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "knowledge_agent": "knowledge_agent",
            "operations_agent": "operations_agent",
            "ticket_agent": "ticket_agent",
            "research_agent": "research_agent",
            "end": "finalizer",
        },
    )

    # ==============================================================
    # 16. Specialist Outcome
    #
    # specialist_router:
    #
    #     failed
    #         → recovery
    #
    #     completed +
    #     workflow_complete=False
    #         → supervisor
    #
    #     completed +
    #     workflow_complete=True
    #         → specialist_completed
    # ==============================================================

    specialist_routes = {
        "recovery": "recovery",
        "supervisor": "supervisor",
        "completed": "specialist_completed",
    }

    for specialist_name in SPECIALIST_NAMES:

        graph.add_conditional_edges(
            specialist_name,
            specialist_router,
            specialist_routes,
        )

    # ==============================================================
    # 17. Specialist Completed
    #
    # 这里决定：
    #
    #     Memory Persist
    #          或
    #     Finalizer
    # ==============================================================

    graph.add_conditional_edges(
        "specialist_completed",
        _workflow_completion_router,
        {
            "memory_persist": (
                "memory_persist"
            ),
            "finalizer": "finalizer",
        },
    )

    # ==============================================================
    # 18. Recovery Router
    #
    # retry:
    #     → 原 Specialist
    #
    # reroute:
    #     → Supervisor
    #
    # human_review:
    #     → Human Review
    #
    # failed:
    #     → Memory Persist / Finalizer
    # ==============================================================

    recovery_routes: dict[str, str] = {
        "knowledge_agent": "knowledge_agent",
        "operations_agent": "operations_agent",
        "ticket_agent": "ticket_agent",
        "research_agent": "research_agent",
        "supervisor": "supervisor",
        "human_review": "human_review",
    }

    if memory_manager is not None:

        recovery_routes[
            "memory_persist"
        ] = "memory_persist"

    else:

        recovery_routes[
            "memory_persist"
        ] = "finalizer"

    graph.add_conditional_edges(
        "recovery",
        recovery_router,
        recovery_routes,
    )

    # ==============================================================
    # 19. Human Review Router
    #
    # human_retry:
    #     → 原 Specialist
    #
    # human_reroute:
    #     → Supervisor
    #
    # human_rejected:
    #     → Memory Persist / Finalizer
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

    else:

        human_review_routes[
            "memory_persist"
        ] = "finalizer"

    graph.add_conditional_edges(
        "human_review",
        human_review_router,
        human_review_routes,
    )

    # ==============================================================
    # 20. Memory Persist → Finalizer
    # ==============================================================

    if memory_manager is not None:

        graph.add_edge(
            "memory_persist",
            "finalizer",
        )

    # ==============================================================
    # 21. Finalizer → END
    # ==============================================================

    graph.add_edge(
        "finalizer",
        END,
    )

    # ==============================================================
    # 22. Compile
    # ==============================================================

    compiled_graph = graph.compile(
        checkpointer=checkpointer,
    )

    return compiled_graph