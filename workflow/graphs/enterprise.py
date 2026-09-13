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
from workflow.routing.recovery_router import recovery_router
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

    注意：
        Specialist Node 使用 LangGraph 的运行时依赖注入，
        签名必须显式保留：

            state
            runtime
            config

        不能使用 *args / **kwargs 替代，
        否则 LangGraph 无法识别并注入 runtime/config。

    Wrapper 只负责：

        Specialist 成功
            ↓
        记录 completed_agents

    Specialist 本身负责：

        Tool Calling
        Tool HITL
        Tool Retry
        Tool Error
        Specialist Failure
    """

    async def wrapped_node(
        state: EnterpriseAgentState,
        runtime: Runtime[EnterpriseAgentContext],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        # ==========================================================
        # 1. 调用原始 Specialist Node
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
        # 3. 获取 Specialist 执行状态
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

        return result

    return wrapped_node


# ==============================================================
# Workflow Completion Router
# ==============================================================

def _workflow_completion_router(
    state: EnterpriseAgentState,
    *,
    memory_enabled: bool,
) -> str:
    """
    Workflow 已经确认完成后的最终路由。

    workflow_complete=True:

        memory_enabled=True
            → memory_persist

        memory_enabled=False
            → finalizer
    """

    if state.get(
        "workflow_complete",
        False,
    ) is not True:
        return "finalizer"

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

    Overall:

        START
          ↓
        Memory Retrieval
          ↓
        Supervisor
          ↓
        Specialist
          ↓
        Specialist Router
          │
          ├── failed
          │      ↓
          │   Recovery
          │
          ├── completed +
          │   workflow_complete=False
          │      ↓
          │   Supervisor
          │
          └── completed +
              workflow_complete=True
                  ↓
              Specialist Completed
                  ↓
              Memory Persist
                  ↓
               Finalizer
                  ↓
                 END


    Supervisor end:

        Supervisor
            ↓
        specialist_completed
            ↓
        memory_persist
            ↓
        finalizer
            ↓
        END

    这样确保：
        Supervisor 最终决定 end 后，
        Long-term Memory Persist 不会被绕过。
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
    # 13. Specialist Completed Intermediate Node
    # ==============================================================

    async def specialist_completed_node(
        state: EnterpriseAgentState,
    ) -> dict[str, Any]:
        """
        Workflow 已经由 Supervisor 确认完成。

        该节点本身不修改业务 State，
        仅作为：

            Supervisor
                ↓
            specialist_completed
                ↓
            Memory / Finalizer

        的显式 Workflow Barrier。
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
    # ==============================================================

    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "knowledge_agent": "knowledge_agent",
            "operations_agent": "operations_agent",
            "ticket_agent": "ticket_agent",
            "research_agent": "research_agent",

            # ======================================================
            # IMPORTANT:
            #
            # Supervisor 的 end 不能直接 Finalizer。
            #
            # 必须经过:
            #
            # supervisor
            #     ↓
            # specialist_completed
            #     ↓
            # memory_persist
            #     ↓
            # finalizer
            #
            # 否则最终 Workflow 会绕过 Long-term Memory Persist。
            # ======================================================
            "end": "specialist_completed",
        },
    )

    # ==============================================================
    # 16. Specialist Outcome Router
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
    # ==============================================================

    graph.add_conditional_edges(
        "specialist_completed",
        lambda state: _workflow_completion_router(
            state,
            memory_enabled=memory_enabled,
        ),
        {
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
    }

    if memory_enabled:

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
    # ==============================================================

    human_review_routes: dict[str, str] = {
        "knowledge_agent": "knowledge_agent",
        "operations_agent": "operations_agent",
        "ticket_agent": "ticket_agent",
        "research_agent": "research_agent",
        "supervisor": "supervisor",
    }

    if memory_enabled:

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

