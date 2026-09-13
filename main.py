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


# ==============================================================
# Workflow Result
# ==============================================================

def _print_workflow_result(
    state: dict[str, Any],
) -> None:
    """
    打印最终 Workflow State。

    最终用户答案只能来自：

        state["final_answer"]

    不再直接使用：

        messages[-1]
    """

    print(
        "\n========== Workflow Result =========="
    )

    # ----------------------------------------------------------
    # Agent / Workflow
    # ----------------------------------------------------------

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
    # HITL
    # ----------------------------------------------------------

    print(
        "Approval Required:",
        state.get(
            "approval_required",
            False,
        ),
    )

    print(
        "Approval Status:",
        state.get(
            "approval_status"
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
    # Tool Execution
    # ----------------------------------------------------------

    print(
        "Tool Results:",
        state.get(
            "tool_results",
            [],
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
    # Execution Truth
    # ----------------------------------------------------------

    print(
        "\n========== Execution Summary =========="
    )

    print(
        state.get(
            "execution_summary",
            {},
        )
    )

    # ----------------------------------------------------------
    # Final Answer
    # ----------------------------------------------------------

    print(
        "\n========== Final Answer =========="
    )

    final_answer = state.get(
        "final_answer"
    )

    if final_answer:
        print(
            final_answer
        )
    else:
        print(
            "Workflow 已结束，但 Finalizer 未生成最终回答。"
        )


# ==============================================================
# Tool-level HITL Resume
# ==============================================================

async def _resume_tool_hitl(
    *,
    graph: Any,
    response: Any,
    config: dict[str, Any],
    context: EnterpriseAgentContext,
) -> Any:
    """
    处理 Tool-level HITL。

    本地测试：

        interrupt
            ↓
        input()
            ↓
        Command(resume=...)
            ↓
        同一个 thread_id resume

    生产环境：

        API
          ↓
        Frontend
          ↓
        Human Decision
          ↓
        /resume API
    """

    while response.interrupts:

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

        # ======================================================
        # Workflow-level Recovery HITL
        # ======================================================

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

            # --------------------------------------------------
            # Workflow Recovery Resume
            #
            # 注意：
            # approval_status 不应该被当作 Recovery
            # 的审批状态，所以这里不修改 approval_status。
            # --------------------------------------------------

            response = await graph.ainvoke(
                Command(
                    resume={
                        "action": decision,
                    },
                    update={
                        "resume_required": False,
                        "task_status": "running",
                    },
                ),
                config=config,
                context=context,
                version="v2",
            )

            continue

        # ======================================================
        # Tool-level HITL
        # ======================================================

        print(
            "\nThis is a Tool-level HITL interrupt."
        )

        action_requests = (
            interrupt_value.get(
                "action_requests",
                [],
            )
            if isinstance(
                interrupt_value,
                dict,
            )
            else []
        )

        review_configs = (
            interrupt_value.get(
                "review_configs",
                [],
            )
            if isinstance(
                interrupt_value,
                dict,
            )
            else []
        )

        if not action_requests:

            print(
                "No action requests found in "
                "the HITL interrupt."
            )

            return response

        # ------------------------------------------------------
        # Display Tool Approval
        # ------------------------------------------------------

        print(
            "\n========== Tool Approval =========="
        )

        for index, action in enumerate(
            action_requests
        ):

            print(
                f"\nAction #{index + 1}:"
            )

            print(
                "Tool:",
                action.get(
                    "name"
                ),
            )

            print(
                "Args:",
                action.get(
                    "args"
                ),
            )

            print(
                "Description:",
                action.get(
                    "description"
                ),
            )

            if index < len(
                review_configs
            ):

                print(
                    "Allowed decisions:",
                    review_configs[
                        index
                    ].get(
                        "allowed_decisions",
                        [],
                    ),
                )

        # ------------------------------------------------------
        # Human Decision
        # ------------------------------------------------------

        while True:

            decision = input(
                "\nTool decision "
                "[approve/edit/reject]: "
            ).strip().lower()

            if decision in {
                "approve",
                "edit",
                "reject",
            }:
                break

            print(
                "Invalid decision. "
                "Please enter approve, "
                "edit, or reject."
            )

        # ======================================================
        # Approve
        # ======================================================

        if decision == "approve":

            decisions = [
                {
                    "type": "approve",
                }
                for _ in action_requests
            ]

            approval_status = (
                "approved"
            )

        # ======================================================
        # Reject
        # ======================================================

        elif decision == "reject":

            decisions = [
                {
                    "type": "reject",
                }
                for _ in action_requests
            ]

            approval_status = (
                "rejected"
            )

        # ======================================================
        # Edit
        #
        # 当前仍使用原 args 验证 edit resume 链路。
        # 后续再扩展成交互式编辑。
        # ======================================================

        else:

            decisions = []

            for action in action_requests:

                current_args = action.get(
                    "args",
                    {},
                )

                print(
                    "\nCurrent tool arguments:"
                )

                print(
                    current_args
                )

                decisions.append(
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": action.get(
                                "name"
                            ),
                            "args": current_args,
                        },
                    }
                )

            approval_status = (
                "edited"
            )

        # ======================================================
        # Resume SAME thread
        # ======================================================

        response = await graph.ainvoke(
            Command(
                resume={
                    "decisions": decisions,
                },
                update={
                    "approval_required": True,
                    "approval_status":
                        approval_status,
                },
            ),
            config=config,
            context=context,
            version="v2",
        )

    return response


# ==============================================================
# Main
# ==============================================================

async def main() -> None:

    # ==========================================================
    # 1. Model
    # ==========================================================

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

    # ==========================================================
    # 2. Runtime Context
    # ==========================================================

    context = EnterpriseAgentContext(
        user_id="user_001",
        user_role="developer",
        tenant_id=None,
    )

    # ==========================================================
    # 3. Execution Config
    # ==========================================================

    config = get_config(
        "user_001"
    )

    # ==========================================================
    # 4. Tool Registry
    # ==========================================================

    async with create_tool_registry() as registry:

        # ======================================================
        # 5. Permission Policy
        # ======================================================

        permission_policy = (
            PermissionPolicy()
        )

        # ======================================================
        # 6. Risk Policy
        # ======================================================

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

        # ======================================================
        # 7. Tool-level HITL
        # ======================================================

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

        # ======================================================
        # 8. Shared Middleware
        #
        #     DynamicToolMiddleware
        #            ↓
        #     Logging / PII
        #            ↓
        #     Tool HITL
        #            ↓
        #     ToolError
        #            ↓
        #     ToolRetry
        #            ↓
        #        Tool
        # ======================================================

        agent_middleware = [
            LoggingMiddleware(),
            *create_pii_middlewares(),
            human_in_the_loop,
            tool_error_async,
            retry_tool_async,
            retry_model_async,
        ]

        # ======================================================
        # 9. Long-term Memory
        # ======================================================

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

        # ======================================================
        # 10. PostgreSQL Checkpointer
        # ======================================================

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

            # ==================================================
            # 11. Build Graph
            # ==================================================

            graph = build_enterprise_graph(
                model=model,
                registry=registry,
                permission_policy=permission_policy,
                middleware=agent_middleware,
                memory_manager=memory_manager,
                checkpointer=checkpointer,
            )

            # ==================================================
            # 12. Initial State
            # ==================================================

            initial_state = {
                # ------------------------------------------------
                # Conversation
                # ------------------------------------------------

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

                # ------------------------------------------------
                # Memory
                # ------------------------------------------------

                "retrieved_memories": [],

                "memory_persisted": False,

                "memory_operation": None,

                "memory_persist_reason": None,

                # ------------------------------------------------
                # Agent / Workflow
                # ------------------------------------------------

                "current_agent": None,

                "next_agent": None,

                "task_status": "running",

                # ------------------------------------------------
                # Routing
                # ------------------------------------------------

                "handoff_reason": None,

                # ------------------------------------------------
                # Tool
                # ------------------------------------------------

                "tool_results": [],

                # ------------------------------------------------
                # HITL
                # ------------------------------------------------

                "approval_required": False,

                "approval_status": None,

                # ------------------------------------------------
                # Error
                # ------------------------------------------------

                "error": None,

                "last_failed_node": None,

                "last_failed_tool": None,

                # ------------------------------------------------
                # Recovery
                # ------------------------------------------------

                "recovery_attempts": 0,

                "recovery_status": None,

                "recovery_reason": None,

                "resume_required": False,

                # ------------------------------------------------
                # Finalizer
                # ------------------------------------------------

                "final_answer": None,

                "execution_summary": {},
            }

            # ==================================================
            # 13. Initial Run
            # ==================================================

            response = await graph.ainvoke(
                initial_state,
                config=config,
                context=context,
                version="v2",
            )

            # ==================================================
            # 14. HITL Resume
            # ==================================================

            if response.interrupts:

                response = await _resume_tool_hitl(
                    graph=graph,
                    response=response,
                    config=config,
                    context=context,
                )

            # ==================================================
            # 15. Final State
            # ==================================================

            final_state = response.value

            # ==================================================
            # 16. Print
            # ==================================================

            _print_workflow_result(
                final_state
            )

        # ======================================================
        # 17. Cleanup
        # ======================================================

        await memory_manager.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )