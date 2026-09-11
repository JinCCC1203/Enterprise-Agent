from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ToolCallRequest,
)
from langchain_openai import ChatOpenAI

from memories.short_memory import checkpointer

from middlewares.LoggingMiddleware import LoggingMiddleware
from middlewares.RetryMiddleware import retry_model_async, retry_tool_async
from middlewares.ToolErrorMiddleware import tool_error_async
from middlewares.dynamic_tools import DynamicToolMiddleware
from middlewares.MemoryMiddleware import (
    MemoryRetrievalMiddleware,
    MemoryPersistenceMiddleware,
)
from middlewares.pii import create_pii_middlewares

from policies.permission import PermissionPolicy
from policies.RiskPolicy import RiskPolicy

from memories.long_memory.embedder import MemoryEmbedder
from memories.long_memory.extractor import MemoryExtractor
from memories.long_memory.manager import MemoryManager
from memories.long_memory.store import MemoryStore

from tools_manager.registration import create_tool_registry
from tools_manager.tool_exposure import PermissionBasedToolExposure
from workflow.state import EnterpriseAgentContext


load_dotenv()


@asynccontextmanager
async def create_agent_app():
    """
    创建 Enterprise-Agent Runtime。

    当前阶段：
        LangChain create_agent + Middleware Runtime

    Runtime 流程：

        Runtime Context
              ↓
        Tool Exposure
              ↓
        Permission Policy
              ↓
        LLM
              ↓
        Tool Calling
              ↓
        Risk Policy
              ↓
        Human-in-the-Loop
              ↓
        Tool Execution

    同时提供：

        - Short-term Memory / Checkpoint
        - Long-term Memory
        - PII Protection
        - Logging
        - Model Retry
        - Tool Retry
        - Tool Error Handling

    后续进入 LangGraph 后：
        Long-term Memory Retrieval / Persistence
        将逐步迁移到 Workflow Nodes。
    """

    model = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0,
    )

    system_prompt = """
你是一个企业级 AI Agent。

你可以使用以下工具：

1. RAG Tool
   用于查询企业内部知识库。

2. MCP Tools
   用于调用外部系统提供的能力。

请根据用户问题自主判断是否需要调用工具。

如果问题涉及企业内部知识，
使用 RAG Tool。

如果需要实时系统状态、工单或企业外部业务能力，
使用对应的 MCP Tool。

不要编造工具没有返回的信息。

如果工具执行失败，
根据错误信息判断是否需要重试或采取其他措施。
"""

    async with create_tool_registry() as registry:

        permission_policy = PermissionPolicy()

        tool_exposure = PermissionBasedToolExposure(
            registry=registry,
            permission_policy=permission_policy,
        )

        dynamic_tool_middleware = DynamicToolMiddleware(
            tool_exposure=tool_exposure,
        )

        risk_policy = RiskPolicy()

        def should_interrupt(
            request: ToolCallRequest,
        ) -> bool:
            """
            判断当前具体 Tool Call 是否需要人工审批。

            注意：
                Tool Exposure
                    =
                Tool 能不能暴露给 LLM

                Risk Policy
                    =
                LLM 已经决定调用后，
                这个具体调用是否需要 HITL
            """

            result = risk_policy.evaluate(
                request=request,
                registry=registry,
            )

            return result.requires_approval


        human_in_the_loop = HumanInTheLoopMiddleware(
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

        # Long-term Memory
        #
        # 当前阶段仍然通过 Middleware 接入。
        # 后续迁移到 LangGraph 后：
        #
        #     Memory Retrieval
        #          ↓
        #     Graph State
        #          ↓
        #     Agent Workflow
        #          ↓
        #     Memory Persistence
        #
        # 不需要重写 MemoryManager / Store / Embedder。
        '''
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

        await memory_manager.initialize()

        memory_retrieval = MemoryRetrievalMiddleware(
            memory_manager=memory_manager,
            top_k=5,
        )

        memory_persistence = MemoryPersistenceMiddleware(
            memory_manager=memory_manager,
        )
        '''

        pii_middlewares = create_pii_middlewares()

        agent = create_agent(
            model=model,

            # Registry 中统一提供：
            #
            #     rag_search
            #     get_service_health
            #     get_ticket
            #     create_ticket
            #     update_ticket
            #     send_notification
            #
            tools=registry.get_all_tools(),

            system_prompt=system_prompt,

            context_schema=EnterpriseAgentContext,

            # Short-term conversation checkpoint
            checkpointer=checkpointer,

            middleware=[
                # Long-term Memory Retrieval
                # 当前 create_agent 阶段：
                # 每次 Model Call 前执行 Memory Retrieval。
                # LangGraph 阶段：
                # 后续迁移为 Workflow-level Memory Retrieval Node。

                # memory_retrieval,

                # Dynamic Tool Exposure
                #
                # Runtime Context
                #      ↓
                # user_role
                #      ↓
                # PermissionPolicy
                #      ↓
                # allowed tools
                #      ↓
                # request.override(tools=...)
                #      ↓
                # LLM Tool Calling
                # ==========================================================

                dynamic_tool_middleware,

                # ==========================================================
                # Logging
                # ==========================================================

                LoggingMiddleware(),

                # ==========================================================
                # PII Protection
                # ==========================================================

                *pii_middlewares,

                # ==========================================================
                # Human-in-the-Loop
                # ==========================================================

                human_in_the_loop,

                # ==========================================================
                # Long-term Memory Persistence
                # 当前阶段：
                # Agent Run 后提取长期记忆。
                # 后续 LangGraph：
                # Workflow-level Memory Persistence Node。
                # ==========================================================

                # memory_persistence,

                # ==========================================================
                # Reliability
                # ==========================================================

                retry_model_async,
                retry_tool_async,
                tool_error_async,
            ],
        )

        yield agent

        # ==================================================================
        # 11. Yield Agent
        # ==================================================================
        '''
        try:
            yield agent

        finally:
            # ==============================================================
            # Cleanup Long-term Memory Resources
            # ==============================================================

            await memory_manager.close()
        '''