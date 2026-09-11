from __future__ import annotations

import asyncio

from agents.single_agent import create_agent_app
from memories.short_memory import get_config
from workflow.graphs.single_agent import build_single_agent_graph
from workflow.state import EnterpriseAgentContext


async def main():
    # --------------------------------------------------------------
    # LangGraph execution config
    #
    # 主要用于：
    # - thread_id
    # - checkpoint
    # - execution configuration
    # --------------------------------------------------------------

    config = get_config(
        "user_001"
    )

    # --------------------------------------------------------------
    # 当前请求的 Runtime Context
    # --------------------------------------------------------------

    context = EnterpriseAgentContext(
        user_id="user_001",
        user_role="developer",
        tenant_id=None,
    )

    # --------------------------------------------------------------
    # 创建现有 LangChain Agent Runtime
    # --------------------------------------------------------------

    async with create_agent_app() as agent:

        # ----------------------------------------------------------
        # 构建 LangGraph
        # ----------------------------------------------------------

        graph = build_single_agent_graph(
            agent=agent,
        )

        # ----------------------------------------------------------
        # 执行 LangGraph
        # ----------------------------------------------------------

        response = await graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "请检查一下 payment-service 的状态"
                        ),
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

        # ----------------------------------------------------------
        # 输出最终消息
        # ----------------------------------------------------------

        print(
            response["messages"][-1].content
        )


if __name__ == "__main__":
    asyncio.run(main())