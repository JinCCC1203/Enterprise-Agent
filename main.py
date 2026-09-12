from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ToolCallRequest,
)
from langchain_openai import ChatOpenAI

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
from langchain_core.messages import HumanMessage


load_dotenv()


async def main() -> None:

    # ==============================================================
    # 1. Model
    # ==============================================================

    model = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=os.getenv(
            "DEEPSEEK_API_KEY"
        ),
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
    # 3. LangGraph execution config
    # ==============================================================

    config = get_config(
        "user_001"
    )

    # ==============================================================
    # 4. Unified Tool Registry
    #
    # MCP Session 在这个 context 中保持存活。
    # 所有 Specialist Agent 都必须在这个 context 中运行。
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
        # ==========================================================

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
        # 7. Human-in-the-Loop
        # ==========================================================

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
        # 注意：
        # DynamicToolMiddleware 不在这里。
        #
        # 每个 Specialist Agent 会根据自身
        # ToolRegistryView 单独创建 DynamicToolMiddleware。
        # ==========================================================

        agent_middleware = [
            LoggingMiddleware(),
            *create_pii_middlewares(),
            human_in_the_loop,
            retry_model_async,
            retry_tool_async,
            tool_error_async,
        ]

        # ==========================================================
        # 9. Enterprise Graph
        # ==========================================================

        graph = build_enterprise_graph(
            model=model,
            registry=registry,
            permission_policy=permission_policy,
            middleware=agent_middleware,
            memory_manager=None,
        )

        # ==========================================================
        # 10. Initial Graph State
        # ==========================================================

        initial_state = {
             "messages": [
        HumanMessage(
            content="请检查一下 payment-service 的状态"
        )
            ],
            "retrieved_memories": [],
            "current_agent": None,
            "next_agent": None,
            "task_status": "running",
            "handoff_reason": None,
            "tool_results": [],
            "approval_required": False,
            "approval_status": None,
            "error": None,
            "retry_count": 0,
            "final_answer": None,
        }

        # ==========================================================
        # 11. Execute Enterprise Workflow
        # ==========================================================

        response = await graph.ainvoke(
            initial_state,
            config=config,
            context=context,
        )

        # ==========================================================
        # 12. Output
        # ==========================================================

        print("\n========== Workflow Result ==========")

        print(
            "Selected Agent:",
            response.get(
                "current_agent"
            ),
        )

        print(
            "Task Status:",
            response.get(
                "task_status"
            ),
        )

        print(
            "Routing Reason:",
            response.get(
                "handoff_reason"
            ),
        )

        messages = response.get(
            "messages",
            [],
        )

        if not messages:
            print(
                "Agent 未返回消息。"
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


if __name__ == "__main__":
    asyncio.run(
        main()
    )