from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from dotenv import load_dotenv

from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ToolCallRequest,
)
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,
)
from langgraph.types import Command

from memories.long_memory.embedder import (
    MemoryEmbedder,
)
from memories.long_memory.extractor import (
    MemoryExtractor,
)
from memories.long_memory.manager import (
    MemoryManager,
)
from memories.long_memory.store import (
    MemoryStore,
)
from memories.short_memory import (
    get_config,
)

from middlewares.LoggingMiddleware import (
    LoggingMiddleware,
)
from middlewares.RetryMiddleware import (
    retry_model_async,
    retry_tool_async,
)
from middlewares.ToolErrorMiddleware import (
    tool_error_async,
)
from middlewares.pii import (
    create_pii_middlewares,
)

from policies.permission import (
    PermissionPolicy,
)
from policies.RiskPolicy import (
    RiskPolicy,
)

from tools_manager.registration import (
    create_tool_registry,
)

from workflow.graphs.enterprise import (
    build_enterprise_graph,
)

from workflow.state import (
    EnterpriseAgentContext,
)


# ==============================================================
# Environment
# ==============================================================

load_dotenv()


# ==============================================================
# Windows asyncio compatibility
# ==============================================================

if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )


def _print_workflow_result(
    state: dict[str, Any],
) -> None:
    """
    打印最终 Workflow State。
    """

    print(
        "\n========== Workflow Result =========="
    )

    print(
        "Selected Agent:",
        state.get(
            "current_agent"
        ),
    )

    print(
        "Task Status:",
        state.get(
            "task_status"
        ),
    )

    print(
        "Routing Reason:",
        state.get(
            "handoff_reason"
        ),
    )

    # ----------------------------------------------------------
    # Memory
    # ----------------------------------------------------------

    print(
        "Retrieved Memories:",
        state.get(
            "retrieved_memories",
            [],
        ),
    )

    print(
        "Memory Persisted:",
        state.get(
            "memory_persisted"
        ),
    )

    print(
        "Memory Operation:",
        state.get(
            "memory_operation"
        ),
    )

    print(
        "Memory Persist Reason:",
        state.get(
            "memory_persist_reason"
        ),
    )

    # ----------------------------------------------------------
    # Recovery
    # ----------------------------------------------------------

    print(
        "Recovery Attempts:",
        state.get(
            "recovery_attempts",
            0,
        ),
    )

    print(
        "Recovery Status:",
        state.get(
            "recovery_status"
        ),
    )

    print(
        "Recovery Reason:",
        state.get(
            "recovery_reason"
        ),
    )

    print(
        "Resume Required:",
        state.get(
            "resume_required",
            False,
        ),
    )

    # ----------------------------------------------------------
    # Error
    # ----------------------------------------------------------

    error = state.get(
        "error"
    )

    if error:
        print(
            "Workflow Error:",
            error,
        )

    print(
        "Last Failed Node:",
        state.get(
            "last_failed_node"
        ),
    )

    print(
        "Last Failed Tool:",
        state.get(
            "last_failed_tool"
        ),
    )

    # ----------------------------------------------------------
    # Final Answer
    # ----------------------------------------------------------

    messages = state.get(
        "messages",
        [],
    )

    if not messages:
        print(
            "\nAgent 未返回消息。"
        )
        return

    final_message = messages[-1]

    content = getattr(
        final_message,
        "content",
        "",
    )

    print(
        "\n========== Final Answer =========="
    )

    print(
        content
    )


async def main() -> None:

    # ==============================================================
    # 1. Model
    # ==============================================================

    deepseek_api_key = os.getenv(
        "DEEPSEEK_API_KEY"
    )

    if not deepseek_api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not configured."
        )

    model = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com",
        temperature=0,
    )

    # ==============================================================
    # 2. Runtime Context
    # ==============================================================

    context = EnterpriseAgentContext(
        user_id="user_001",
        user_role="developer",
        tenant_id=None,
    )

    # ==============================================================
    # 3. LangGraph Execution Config
    #
    # 必须包含：
    #
    #     configurable.thread_id
    #
    # thread_id 是：
    #
    #     Checkpoint
    #     Interrupt
    #     Resume
    #     Recovery
    #
    # 的持久化标识。
    # ==============================================================

    config = get_config(
        "user_001"
    )

    # ==============================================================
    # 4. Unified Tool Registry
    # ==============================================================

    async with create_tool_registry() as registry:

        # ==========================================================
        # 5. Permission Policy
        # ==========================================================

        permission_policy = (
            PermissionPolicy()
        )

        # ==========================================================
        # 6. Risk Policy
        # ==============================================================

        risk_policy = RiskPolicy()

        def should_interrupt(
            request: ToolCallRequest,
        ) -> bool:

            result = (
                risk_policy.evaluate(
                    request=request,
                    registry=registry,
                )
            )

            return result.requires_approval

        # ==========================================================
        # 7. Tool-level Human-in-the-Loop
        #
        # 注意：
        #
        # 这是 Tool-level HITL。
        #
        # 与 workflow/nodes/human_review.py
        # 的 Workflow-level Recovery HITL 不同。
        # ==============================================================

        human_in_the_loop = (
            HumanInTheLoopMiddleware(
                interrupt_on={
                    tool.name: {
                        "allowed_decisions": [
                            "approve",
                            "edit",
                            "reject",
                        ],
                        "when": should_interrupt,
                    }
                    for tool in registry.get_all_tools()
                }
            )
        )

        # ==========================================================
        # 8. Shared Middleware
        #
        # DynamicToolMiddleware 不放这里。
        #
        # 每个 Specialist：
        #
        #     DynamicToolMiddleware
        #         ↓
        #     shared middleware
        #
        # Tool Error 位于 Tool Retry 外层：
        #
        #     ToolError
        #         ↓
        #     ToolRetry
        #         ↓
        #        Tool
        # ==============================================================

        agent_middleware = [
            LoggingMiddleware(),

            *create_pii_middlewares(),

            human_in_the_loop,

            tool_error_async,

            retry_tool_async,

            retry_model_async,
        ]

        # ==========================================================
        # 9. Long-term Memory
        # ==============================================================

        memory_database_url = os.getenv(
            "MEMORY_DATABASE_URL"
        )

        if not memory_database_url:
            raise ValueError(
                "MEMORY_DATABASE_URL is not configured."
            )

        memory_store = MemoryStore(
            database_url=memory_database_url,
        )

        memory_embedder = MemoryEmbedder(
            model_name="BAAI/bge-base-en-v1.5",
            expected_dim=768,
        )

        memory_extractor = MemoryExtractor(
            model=model,
        )

        memory_manager = MemoryManager(
            extractor=memory_extractor,
            embedder=memory_embedder,
            store=memory_store,
            min_confidence=0.7,
            similarity_threshold=0.3,
            duplicate_threshold=0.1,
        )

        await memory_manager.initialize()

        # ==========================================================
        # 10. LangGraph PostgreSQL Checkpointer
        # ==============================================================

        langgraph_database_url = os.getenv(
            "LANGGRAPH_DATABASE_URL"
        )

        if not langgraph_database_url:
            raise ValueError(
                "LANGGRAPH_DATABASE_URL is not configured."
            )

        async with AsyncPostgresSaver.from_conn_string(
            langgraph_database_url,
        ) as checkpointer:

            await checkpointer.setup()

            # ======================================================
            # 11. Build Enterprise Graph
            # ======================================================

            graph = build_enterprise_graph(
                model=model,
                registry=registry,
                permission_policy=permission_policy,
                middleware=agent_middleware,
                memory_manager=memory_manager,
                checkpointer=checkpointer,
            )

            # ======================================================
            # 12. Initial Graph State
            # ======================================================

            initial_state = {
                # --------------------------------------------------
                # Conversation
                # --------------------------------------------------

                "messages": [
                    HumanMessage(
                        content=(
                            "请为 payment-service 创建一个 P1 Incident 工单。"
                            "标题为“payment-service 高延迟告警”，"
                            "描述为“当前 payment-service 出现严重延迟，"
                            "需要立即排查并通知相关团队。”"
                            "如果系统要求人工审批，请暂停并等待我的确认。"
                        )
                    )
                ],

                # --------------------------------------------------
                # Long-term Memory
                # --------------------------------------------------

                "retrieved_memories": [],

                "memory_persisted": False,

                "memory_operation": None,

                "memory_persist_reason": None,

                # --------------------------------------------------
                # Agent / Workflow
                # --------------------------------------------------

                "current_agent": None,

                "next_agent": None,

                "task_status": "running",

                # --------------------------------------------------
                # Routing / Handoff
                # --------------------------------------------------

                "handoff_reason": None,

                # --------------------------------------------------
                # Tool Results
                # --------------------------------------------------

                "tool_results": [],

                # --------------------------------------------------
                # HITL
                # --------------------------------------------------

                "approval_required": False,

                "approval_status": None,

                # --------------------------------------------------
                # Error
                # --------------------------------------------------

                "error": None,

                "last_failed_node": None,

                "last_failed_tool": None,

                # --------------------------------------------------
                # Workflow Recovery
                # --------------------------------------------------

                "recovery_attempts": 0,

                "recovery_status": None,

                "recovery_reason": None,

                "resume_required": False,

                # --------------------------------------------------
                # Final Answer
                # --------------------------------------------------

                "final_answer": None,
            }

            # ======================================================
            # 13. Initial Workflow Execution
            #
            # version="v2":
            #
            #     response.value
            #     response.interrupts
            #
            # 当前 LangGraph 推荐的类型安全接口。
            # ==============================================================

            response = await graph.ainvoke(
                initial_state,
                config=config,
                context=context,
                version="v2",
            )

            # ======================================================
            # 14. Workflow-level Human Review
            # ==============================================================

            if response.interrupts:

                interrupt_value = (
                    response.interrupts[0].value
                )

                print(
                    "\n========== Workflow Interrupted =========="
                )

                print(
                    "Interrupt:",
                    interrupt_value,
                )

                # --------------------------------------------------
                # 当前用于本地端到端测试。
                #
                # 真实生产环境这里应该：
                #
                #     返回 API
                #         ↓
                #     前端展示
                #         ↓
                #     用户决定
                #         ↓
                #     /resume API
                #         ↓
                #     Command(resume=...)
                #
                # 当前先用 input() 模拟人工决策。
                # --------------------------------------------------

                if (
                    isinstance(
                        interrupt_value,
                        dict,
                    )
                    and interrupt_value.get(
                        "type"
                    )
                    == "workflow_recovery_review"
                ):

                    while True:

                        decision = input(
                            "\nRecovery decision "
                            "[retry/reroute/reject]: "
                        ).strip().lower()

                        if decision in {
                            "retry",
                            "reroute",
                            "reject",
                        }:
                            break

                        print(
                            "Invalid decision. "
                            "Please enter retry, "
                            "reroute, or reject."
                        )

                    # ------------------------------------------------
                    # Resume the SAME thread.
                    #
                    # Command(resume=...)
                    # 会成为 interrupt() 的返回值。
                    # ------------------------------------------------

                    response = await graph.ainvoke(
                        Command(
                            resume={
                                "action": decision,
                            }
                        ),
                        config=config,
                        context=context,
                        version="v2",
                    )

                    # ------------------------------------------------
                    # 如果恢复后又触发新的 Workflow-level interrupt，
                    # 可以继续处理。
                    #
                    # 当前示例只处理一次。
                    # ------------------------------------------------

                    while response.interrupts:

                        interrupt_value = (
                            response.interrupts[0].value
                        )

                        print(
                            "\n========== "
                            "Workflow Interrupted Again "
                            "=========="
                        )

                        print(
                            "Interrupt:",
                            interrupt_value,
                        )

                        if not (
                            isinstance(
                                interrupt_value,
                                dict,
                            )
                            and interrupt_value.get(
                                "type"
                            )
                            == "workflow_recovery_review"
                        ):
                            print(
                                "Received a non-recovery "
                                "interrupt. "
                                "External HITL handling "
                                "is required."
                            )
                            return

                        decision = input(
                            "\nRecovery decision "
                            "[retry/reroute/reject]: "
                        ).strip().lower()

                        if decision not in {
                            "retry",
                            "reroute",
                            "reject",
                        }:
                            print(
                                "Invalid decision."
                            )
                            continue

                        response = await graph.ainvoke(
                            Command(
                                resume={
                                    "action": decision,
                                }
                            ),
                            config=config,
                            context=context,
                            version="v2",
                        )

                else:
                    # ------------------------------------------------
                    # Tool-level HITL
                    #
                    # HumanInTheLoopMiddleware 产生的 interrupt
                    # 使用自己的 decision schema。
                    #
                    # 此处暂不自动处理。
                    # ------------------------------------------------

                    print(
                        "This is a Tool-level HITL interrupt."
                    )

                    print(
                        "Use the HumanInTheLoopMiddleware "
                        "decision schema to resume it."
                    )

                    return

            # ======================================================
            # 15. Final Graph State
            #
            # version="v2" → response.value
            # ==============================================================

            final_state = response.value

            _print_workflow_result(
                final_state
            )

        # ==========================================================
        # 16. Cleanup Long-term Memory
        # ==========================================================

        await memory_manager.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )