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
from middlewares.RetryMiddleware import retry_model, retry_tool
from middlewares.ToolErrorMiddleware import tool_error
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
from tools_manager.selectors import RuleBasedToolSelector


load_dotenv()

@asynccontextmanager
async def create_agent_app():
    #配置模型
    model=ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0
    )
    #系统提示词工程
    system_prompt = """
    你是一个企业级 AI Agent。

    你可以使用以下工具：

    1. RAG Tool
       用于查询企业内部知识库。

    2. MCP Tools
       用于调用外部系统提供的能力。

    请根据用户问题自主判断是否需要调用工具。

    如果问题涉及企业内部知识，
    优先使用 RAG Tool。

    如果需要外部系统能力，
    使用 MCP Tool。

    不要编造工具没有返回的信息。
    """

    async with create_tool_registry() as registry:
        permission_policy = PermissionPolicy()

        selector = RuleBasedToolSelector(
            registry=registry,
            permission_policy=permission_policy,
        )

        dynamic_tool_middleware = DynamicToolMiddleware(
            selector=selector,
        )

        risk_policy = RiskPolicy()

        def should_interrupt(
                request: ToolCallRequest,
        ) -> bool:
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

        # Memory Store
        memory_store = MemoryStore(
            database_url=os.getenv("MEMORY_DATABASE_URL"),
        )

        # Memory Embedder
        memory_embedder = MemoryEmbedder(
            model_name="BAAI/bge-base-en-v1.5",
            expected_dim=768,
        )

        # Memory Extractor
        memory_extractor = MemoryExtractor(
            model=model,
        )

        # Memory Manager
        memory_manager = MemoryManager(
            extractor=memory_extractor,
            embedder=memory_embedder,
            store=memory_store,
            min_confidence=0.7,
            similarity_threshold=0.3,
            duplicate_threshold=0.1,
        )

        # 初始化 PostgreSQL + pgvector
        await memory_manager.initialize()

        memory_retrieval = MemoryRetrievalMiddleware(
            memory_manager=memory_manager,
            top_k=5,
        )

        memory_persistence = MemoryPersistenceMiddleware(
            memory_manager=memory_manager,
        )

        pii_middlewares = create_pii_middlewares()


        agent = create_agent(
            model=model,
            tools=registry.get_all_tools(),
            system_prompt=system_prompt,
            checkpointer=checkpointer,

            middleware=[
                # Long-term Memory
                memory_retrieval,

                # Tool Selection
                dynamic_tool_middleware,

                # PII
                *pii_middlewares,

                # Human-in-the-Loop
                human_in_the_loop,

                # Long-term Memory Persistence
                memory_persistence,

                # Logging / Reliability
                LoggingMiddleware(),
                retry_model,
                retry_tool,
                tool_error,
            ],
        )

        yield agent

        await memory_manager.close()




