from __future__ import annotations

import asyncio
import os
import sys
import uuid
from typing import Any

from dotenv import load_dotenv
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ToolCallRequest,
)
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from memories.long_memory.embedder import MemoryEmbedder
from memories.long_memory.extractor import MemoryExtractor
from memories.long_memory.manager import MemoryManager
from memories.long_memory.store import MemoryStore
from memories.short_memory import get_config
from middlewares.LoggingMiddleware import LoggingMiddleware
from middlewares.RetryMiddleware import (
    retry_model_async,
    retry_tool_async,
)
from middlewares.ToolErrorMiddleware import tool_error_async
from middlewares.pii import create_pii_middlewares
from policies.permission import PermissionPolicy
from policies.RiskPolicy import RiskPolicy
from tools_manager.registration import create_tool_registry
from workflow.graphs.enterprise import build_enterprise_graph
from workflow.state import EnterpriseAgentContext


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
# Output Helper
# ==============================================================


def _one_line(
    value: Any,
) -> str:
    """
    将任意输出压缩成单行字符串。
    """

    if value is None:
        return "None"

    text = str(value)

    return " ".join(
        text.split()
    )


# ==============================================================
# Workflow Result
# ==============================================================


def _print_workflow_result(
    state: dict[str, Any],
) -> None:
    """
    打印最终 Workflow State。

    所有输出保持单行。
    """

    print("========== Workflow Result ==========")
    print(
        f"Selected Agent: {_one_line(state.get('current_agent'))}"
    )
    print(
        f"Next Agent: {_one_line(state.get('next_agent'))}"
    )
    print(
        f"Task Status: {_one_line(state.get('task_status'))}"
    )
    print(
        f"Workflow Complete: {_one_line(state.get('workflow_complete', False))}"
    )
    print(
        f"Completed Agents: {_one_line(state.get('completed_agents', []))}"
    )
    print(
        f"Routing Reason: {_one_line(state.get('handoff_reason'))}"
    )
    print(
        f"Approval Required: {_one_line(state.get('approval_required', False))}"
    )
    print(
        f"Approval Status: {_one_line(state.get('approval_status'))}"
    )
    print(
        f"Approval Events: {_one_line(state.get('approval_events', []))}"
    )
    print(
        f"Retrieved Memories: {_one_line(state.get('retrieved_memories', []))}"
    )
    print(
        f"Memory Persisted: {_one_line(state.get('memory_persisted'))}"
    )
    print(
        f"Memory Operation: {_one_line(state.get('memory_operation'))}"
    )
    print(
        f"Memory Persist Reason: {_one_line(state.get('memory_persist_reason'))}"
    )
    print(
        f"Recovery Attempts: {_one_line(state.get('recovery_attempts', 0))}"
    )
    print(
        f"Recovery Status: {_one_line(state.get('recovery_status'))}"
    )
    print(
        f"Recovery Reason: {_one_line(state.get('recovery_reason'))}"
    )
    print(
        f"Resume Required: {_one_line(state.get('resume_required', False))}"
    )
    print(
        f"Tool Results: {_one_line(state.get('tool_results', []))}"
    )
    print(
        f"Workflow Error: {_one_line(state.get('error'))}"
    )
    print(
        f"Last Failed Node: {_one_line(state.get('last_failed_node'))}"
    )
    print(
        f"Last Failed Tool: {_one_line(state.get('last_failed_tool'))}"
    )

    print(
        "========== Execution Summary =========="
    )
    print(
        f"Execution Summary: {_one_line(state.get('execution_summary', {}))}"
    )

    print(
        "========== Final Answer =========="
    )

    final_answer = state.get(
        "final_answer"
    )

    if final_answer:
        print(
            f"Final Answer: {_one_line(final_answer)}"
        )
    else:
        print(
            "Final Answer: Workflow 已结束，但 Finalizer 未生成最终回答。"
        )


# ==============================================================
# Tool-level / Workflow-level HITL Resume
# ==============================================================


async def _resume_tool_hitl(
    *,
    graph: Any,
    response: Any,
    config: dict[str, Any],
    context: EnterpriseAgentContext,
) -> Any:
    """
    处理：

        Tool-level HITL
        Workflow-level Recovery HITL
    """

    while response.interrupts:

        interrupt_value = (
            response.interrupts[0].value
        )

        print(
            "========== Workflow Interrupted =========="
        )
        print(
            f"Interrupt: {_one_line(interrupt_value)}"
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
                    "Recovery decision [retry/reroute/reject]: "
                ).strip().lower()

                if decision in {
                    "retry",
                    "reroute",
                    "reject",
                }:
                    break

                print(
                    "Invalid decision. Please enter retry, reroute, or reject."
                )

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
            "This is a Tool-level HITL interrupt."
        )

        if isinstance(
            interrupt_value,
            dict,
        ):
            action_requests = interrupt_value.get(
                "action_requests",
                [],
            )
            review_configs = interrupt_value.get(
                "review_configs",
                [],
            )
        else:
            action_requests = []
            review_configs = []

        if not action_requests:
            print(
                "No action requests found in the HITL interrupt."
            )
            return response

        print(
            "========== Tool Approval =========="
        )

        for index, action in enumerate(
            action_requests
        ):

            print(
                f"Action #{index + 1}: Tool={_one_line(action.get('name'))}"
            )
            print(
                f"Action #{index + 1}: Args={_one_line(action.get('args'))}"
            )
            print(
                f"Action #{index + 1}: Description={_one_line(action.get('description'))}"
            )

            if index < len(
                review_configs
            ):
                print(
                    f"Action #{index + 1}: Allowed decisions={_one_line(review_configs[index].get('allowed_decisions', []))}"
                )

        # ======================================================
        # Human Decision
        # ======================================================

        while True:

            decision = input(
                "Tool decision [approve/edit/reject]: "
            ).strip().lower()

            if decision in {
                "approve",
                "edit",
                "reject",
            }:
                break

            print(
                "Invalid decision. Please enter approve, edit, or reject."
            )

        # ======================================================
        # Approve
        # ======================================================

        if decision == "approve":

            decisions = [
                {
                    "type": "approve"
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
                    "type": "reject"
                }
                for _ in action_requests
            ]

            approval_status = (
                "rejected"
            )

        # ======================================================
        # Edit
        # ======================================================

        else:

            decisions = []

            for action in action_requests:

                current_args = action.get(
                    "args",
                    {},
                )

                print(
                    f"Current tool arguments: {_one_line(current_args)}"
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
        # Approval Event
        # ======================================================

        approval_event = {
            "type": "tool_approval",
            "tool_names": [
                action.get("name")
                for action in action_requests
                if action.get("name")
            ],
            "decision": approval_status,
        }

        # ======================================================
        # Resume same thread
        # ======================================================

        response = await graph.ainvoke(
            Command(
                resume={
                    "decisions": decisions,
                },
                update={
                    "approval_events": [
                        approval_event
                    ],
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
        extra_body={
            "thinking": {
                "type": "disabled",
            }
        },
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
        "multi-specialist-test-002"
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

            result = risk_policy.evaluate(
                request=request,
                registry=registry,
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
            # 11. Enterprise Graph
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
            # 12. Multi-Specialist Test Query
            # ==================================================

            query = (
                "查询 payment-service 当前健康状态。"
                "如果发现服务异常，请创建一个 P1 Incident 工单。"
            )

            print(
                f"Test Query: {query}"
            )

            # ==================================================
            # 13. Initial State
            # ==================================================

            initial_state = {
                "messages": [
                    HumanMessage(
                        content=query
                    )
                ],
                "retrieved_memories": [],
                "memory_persisted": False,
                "memory_operation": None,
                "memory_persist_reason": None,
                "current_agent": None,
                "next_agent": None,
                "task_status": "running",
                "workflow_complete": False,
                "completed_agents": [],
                "handoff_reason": None,
                "tool_results": [],
                "approval_required": False,
                "approval_status": None,
                "approval_events": [],
                "error": None,
                "last_failed_node": None,
                "last_failed_tool": None,
                "recovery_attempts": 0,
                "recovery_status": None,
                "recovery_reason": None,
                "resume_required": False,
                "final_answer": None,
                "execution_summary": {},
            }

            # ==================================================
            # 14. Initial Execution
            # ==================================================

            response = await graph.ainvoke(
                initial_state,
                config=config,
                context=context,
                version="v2",
            )

            # ==================================================
            # 15. HITL Resume
            # ==================================================

            if response.interrupts:

                response = await _resume_tool_hitl(
                    graph=graph,
                    response=response,
                    config=config,
                    context=context,
                )

            # ==================================================
            # 16. Final State
            # ==================================================

            final_state = response.value

            # ==================================================
            # 17. Result
            # ==================================================

            _print_workflow_result(
                final_state
            )

        # ======================================================
        # 18. Cleanup
        # ======================================================

        await memory_manager.close()


# ==============================================================
# Entry Point
# ==============================================================


if __name__ == "__main__":
    asyncio.run(
        main()
    )