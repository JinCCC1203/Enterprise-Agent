from __future__ import annotations

import asyncio
import os

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

from memories.long_memory.embedder import MemoryEmbedder
from memories.long_memory.extractor import MemoryExtractor
from memories.long_memory.manager import MemoryManager
from memories.long_memory.store import MemoryStore
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
    #
    # get_config() 必须包含：
    #
    #     configurable.thread_id
    #
    # 该 thread_id 是 Checkpoint / Resume 的核心标识。
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
        # DynamicToolMiddleware 不在这里。
        #
        # 每个 Specialist Agent 根据自己的
        # ToolRegistryView 单独创建 DynamicToolMiddleware。
        # ==============================================================

        agent_middleware = [
            LoggingMiddleware(),
            *create_pii_middlewares(),
            human_in_the_loop,
            retry_model_async,
            retry_tool_async,
            tool_error_async,
        ]

        # ==========================================================
        # 9. Long-term Memory
        # ==============================================================

        memory_store = MemoryStore(
            database_url=os.getenv(
                "MEMORY_DATABASE_URL"
            ),
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

        # 初始化长期记忆数据库表
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

        # AsyncPostgresSaver 会自动管理异步连接生命周期。
        async with AsyncPostgresSaver.from_conn_string(
            langgraph_database_url,
        ) as checkpointer:

            # 第一次使用时创建 checkpoint 相关表。
            #
            # setup() 是幂等的：
            # 已存在的 migration 不会重复创建。
            await checkpointer.setup()

            # ======================================================
            # 11. Enterprise Graph
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
            #
            # 使用 HumanMessage，而不是：
            #
            #     {"role": "user", "content": "..."}
            #
            # 因为 Memory Retrieval Node 会从
            # LangChain BaseMessage 中提取当前 Query。
            # ======================================================

            initial_state = {
                "messages": [
                    HumanMessage(
                        content=(
                            "请搜索一下最近 DeepSeek 发布的最新模型，"
                            "并总结其主要更新内容。"
                        )
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

            # ======================================================
            # 13. Execute Enterprise Workflow
            # ======================================================

            response = await graph.ainvoke(
                initial_state,
                config=config,
                context=context,
            )

            # ======================================================
            # 14. Output
            # ======================================================

            print(
                "\n========== Workflow Result =========="
            )

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

            retrieved_memories = response.get(
                "retrieved_memories",
                [],
            )

            print(
                "Retrieved Memories:",
                retrieved_memories,
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

        # ==========================================================
        # 15. Cleanup Long-term Memory
        # ==========================================================

        await memory_manager.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )