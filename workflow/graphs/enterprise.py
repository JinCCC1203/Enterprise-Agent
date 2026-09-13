from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from memories.long_memory.manager import MemoryManager
from policies.permission import PermissionPolicy
from tools_manager.registry import ToolRegistry
from workflow.nodes.finalizer import create_finalizer_node
from workflow.nodes.human_review import create_human_review_node
from workflow.nodes.knowledge_agent import create_knowledge_agent
from workflow.nodes.memory import create_memory_retrieval_node
from workflow.nodes.memory_persist import create_memory_persist_node
from workflow.nodes.operations_agent import create_operations_agent
from workflow.nodes.recovery import (
    RecoveryPolicy,
    create_recovery_node,
)
from workflow.nodes.research_agent import create_research_agent
from workflow.nodes.supervisor import create_supervisor_node
from workflow.nodes.ticket_agent import create_ticket_agent
from workflow.routing.human_review_router import human_review_router
from workflow.routing.specialist_router import specialist_router
from workflow.routing.supervisor_router import supervisor_router
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

    核心原则：

        Specialist completed
            !=
        Workflow completed

    Specialist 只能报告自己的 delegated task 已完成。

    整个 Workflow 是否完成：
        只能由 Supervisor 决定。
    """

    async def wrapped_node(
            state: EnterpriseAgentState,
            runtime: Runtime[EnterpriseAgentContext],
            config: RunnableConfig,
    ) -> dict[str, Any]:

        # ==========================================================
        # 1. 执行原始 Specialist
        # ==========================================================

        result = await node(
            state,
            runtime,
            config,
        )

        # ==========================================================
        # 2. 防御性检查
        # ==========================================================

        if not isinstance(
                result,
                dict,
        ):
            return result

        # ==========================================================
        # 3. 获取 Specialist 子任务状态
        # ==========================================================

        task_status = result.get(
            "task_status",
            state.get(
                "task_status",
                "running",
            ),
        )

        # ==========================================================
        # 4. Specialist 成功
        #
        # IMPORTANT:
        #
        # Specialist completed
        #     !=
        # Workflow completed
        #
        # 因此这里强制：
        #
        # workflow_complete=False
        #
        # 然后回 Supervisor 重新规划。
        # ==========================================================

        if task_status == "completed":

            completed_agents = list(
                state.get(
                    "completed_agents",
                    [],
                )
            )

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

            # ------------------------------------------------------
            # Specialist 无权决定整个 Workflow 完成
            # ------------------------------------------------------

            result["workflow_complete"] = False

            # ------------------------------------------------------
            # Specialist 完成后重新进入 Supervisor
            # ------------------------------------------------------

            result["next_agent"] = "supervisor"

        return result

    return wrapped_node


# ==============================================================
# Specialist Outcome Router
# ==============================================================

def _specialist_outcome_router(
        state: EnterpriseAgentState,
) -> str:
    """
    Specialist 执行结果的 Graph-level Router。

    规则：

        failed
            → recovery

        specialist 自己明确需要 Supervisor
            → supervisor

        completed
            → supervisor

    注意：

        completed 永远不能直接进入 finalizer。

        因为：
            Specialist completed
            !=
            Workflow completed
    """

    route = specialist_router(
        state
    )

    # ==========================================================
    # 1. Recovery
    # ==========================================================

    if route == "recovery":
        return "recovery"

    # ==========================================================
    # 2. Supervisor
    # ==========================================================

    if route == "supervisor":
        return "supervisor"

    # ==========================================================
    # 3. Specialist completed
    #
    # 无条件返回 Supervisor。
    #
    # Supervisor 才决定：
    #
    #     还有后续任务？
    #         ↓
    #     specialist
    #
    #     没有？
    #         ↓
    #     end
    # ==========================================================

    if route == "completed":
        return "supervisor"

    # ==========================================================
    # 4. Defensive fallback
    # ==========================================================

    return "supervisor"


# ==============================================================
# Workflow Completion Router
# ==============================================================

def _workflow_completion_router(
        state: EnterpriseAgentState,
        *,
        memory_enabled: bool,
) -> str:
    """
    Workflow Completion Router。

    只有 Supervisor 明确返回：

        workflow_complete=True

    且：

        next_agent=end

    才应该到达 specialist_completed。

    正常情况下：
        specialist_completed
            → memory_persist / finalizer

    如果异常状态下：
        workflow_complete=False

    则重新回 Supervisor，避免错误结束。
    """

    if state.get(
            "workflow_complete",
            False,
    ) is not True:
        return "supervisor"

    if memory_enabled:
        return "memory_persist"

    return "finalizer"


# ==============================================================
# Recovery Router
# ==============================================================

def _recovery_graph_router(
        state: EnterpriseAgentState,
) -> str:
    recovery_status = state.get(
        "recovery_status"
    )

    next_agent = state.get(
        "next_agent"
    )

    # ==========================================================
    # Terminal Failure
    # ==========================================================

    if recovery_status == "failed":
        return "terminal"

    # ==========================================================
    # Reroute → Supervisor
    # ==========================================================

    if recovery_status == "reroute":
        return "supervisor"

    # ==========================================================
    # Retry Original Specialist
    # ==========================================================

    if recovery_status == "retry":

        if next_agent in SPECIALIST_NAMES:
            return next_agent

        return "terminal"

    # ==========================================================
    # Human Review
    # ==========================================================

    if recovery_status == "human_review":
        return "human_review"

    # ==========================================================
    # Defensive Fallback
    # ==========================================================

    return "terminal"


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

    核心流程：

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
          │      ↓
          │   Re-plan
          │
          └── ...
                 ↓
             Supervisor
                 ↓
          end / next specialist
                 ↓
        specialist_completed
                 ↓
        Memory Persist
                 ↓
             Finalizer
                 ↓
                END
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

    memory_enabled = (
            memory_manager is not None
    )

    # ==============================================================
    # 3. Memory Retrieval
    # ==============================================================

    if memory_enabled:
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
    # 9. Memory Persist
    # ==============================================================

    if memory_enabled:
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

    finalizer_node = create_finalizer_node()

    graph.add_node(
        "finalizer",
        finalizer_node,
    )

    # ==============================================================
    # 13. Specialist Completed Barrier
    # ==============================================================

    async def specialist_completed_node(
            state: EnterpriseAgentState,
    ) -> dict[str, Any]:
        """
        Specialist Completed Barrier。

        正常进入此节点意味着：

            Supervisor 已经明确判断：

                workflow_complete=True
                next_agent=end

        因此这里不再修改业务状态。
        """

        return {}

    graph.add_node(
        "specialist_completed",
        specialist_completed_node,
    )

    # ==============================================================
    # 14. START
    # ==============================================================

    if memory_enabled:

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
    # 15. Supervisor Router
    #
    # Supervisor 才拥有：
    #
    #     workflow_complete=True
    #
    # 的最终决定权。
    # =============================================================

    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "knowledge_agent": "knowledge_agent",
            "operations_agent": "operations_agent",
            "ticket_agent": "ticket_agent",
            "research_agent": "research_agent",
            "end": "specialist_completed",
        },
    )

    # ==============================================================
    # 16. Specialist Outcome Router
    # ==============================================================

    specialist_routes = {
        "recovery": "recovery",
        "supervisor": "supervisor",
        "completed": "supervisor",
    }

    for specialist_name in SPECIALIST_NAMES:
        graph.add_conditional_edges(
            specialist_name,
            _specialist_outcome_router,
            specialist_routes,
        )

    # ==============================================================
    # 17. Specialist Completed Barrier
    # ==============================================================

    graph.add_conditional_edges(
        "specialist_completed",
        lambda state: _workflow_completion_router(
            state,
            memory_enabled=memory_enabled,
        ),
        {
            "supervisor": "supervisor",
            "memory_persist": "memory_persist",
            "finalizer": "finalizer",
        },
    )

    # ==============================================================
    # 18. Recovery Router
    # ==============================================================

    recovery_routes: dict[str, str] = {
        "knowledge_agent": "knowledge_agent",
        "operations_agent": "operations_agent",
        "ticket_agent": "ticket_agent",
        "research_agent": "research_agent",
        "supervisor": "supervisor",
        "human_review": "human_review",
        "terminal": (
            "memory_persist"
            if memory_enabled
            else "finalizer"
        ),
    }

    graph.add_conditional_edges(
        "recovery",
        _recovery_graph_router,
        recovery_routes,
    )

    # ==============================================================
    # 19. Human Review Router
    # ==============================================================

    human_review_routes: dict[str, str] = {
        "knowledge_agent": "knowledge_agent",
        "operations_agent": "operations_agent",
        "ticket_agent": "ticket_agent",
        "research_agent": "research_agent",
        "supervisor": "supervisor",
        "memory_persist": (
            "memory_persist"
            if memory_enabled
            else "finalizer"
        ),
    }

    graph.add_conditional_edges(
        "human_review",
        human_review_router,
        human_review_routes,
    )

    # ==============================================================
    # 20. Memory Persist → Finalizer
    # ==============================================================

    if memory_enabled:
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