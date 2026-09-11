from __future__ import annotations

import asyncio

from agents.single_agent import create_agent_app
from memories.short_memory import get_config
from workflow.graphs.single_agent import build_single_agent_graph
from workflow.state import EnterpriseAgentContext


async def main() -> None:
    # ==============================================================
    # LangGraph execution config
    #
    # 用于：
    #   - thread_id
    #   - checkpoint
    #   - 其他执行配置
    # ==============================================================

    config = get_config("user_001")

    # ==============================================================
    # Runtime Context
    #
    # 用于：
    #   - user_id
    #   - user_role
    #   - tenant_id
    #
    # 注意：
    #   这些信息属于可信运行时上下文，
    #   不属于 Graph State。
    # ==============================================================

    context = EnterpriseAgentContext(
        user_id="user_001",
        user_role="developer",
        tenant_id=None,
    )

    # ==============================================================
    # 创建 LangChain Agent Runtime
    # ==============================================================

    async with create_agent_app() as agent:

        # ==========================================================
        # 构建 LangGraph
        #
        # 当前阶段：
        #
        #   START
        #     ↓
        #   Agent
        #     ↓
        #    END
        #
        # 后续：
        #
        #   START
        #     ↓
        #   Memory Retrieve
        #     ↓
        #   Supervisor
        #     ↓
        #   Specialist
        #     ↓
        #   Memory Persist
        #     ↓
        #    END
        # ==========================================================

        graph = build_single_agent_graph(
            agent=agent,
        )

        # ==========================================================
        # 执行 LangGraph
        # ==========================================================

        response = await graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "请检查一下 payment-service 的状态",
                    }
                ],
                "retrieved_memories": [],
                "current_agent": "single_agent",
                "task_status": "running",
                "tool_results": [],
                "error": None,
                "retry_count": 0,
                "final_answer": None,
            },
            config=config,
            context=context,
        )

        # ==========================================================
        # 输出最终结果
        # ==========================================================

        messages = response.get("messages", [])

        if not messages:
            print("Agent 未返回消息。")
            return

        final_message = messages[-1]

        content = getattr(
            final_message,
            "content",
            "",
        )

        print(content)


if __name__ == "__main__":
    asyncio.run(main())