from tools_manager.registry import ToolRegistry
from tools_manager.metadata import ToolMetadata, ToolRiskLevel, ToolSource
from tools.rag_tool import rag_search
from mcp_client.client import (
mcp_session,
get_mcp_tools
)
from mcp_adapter.mcp_to_langchain import convert_mcp_tools
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def create_tool_registry() -> AsyncIterator[ToolRegistry]:

    registry=ToolRegistry()

    registry.register(
        rag_search,
        ToolMetadata(
            name=rag_search.name,
            category="knowledge",
            permissions=frozenset({"employee", "developer", "admin"}),
            tags=frozenset({"rag", "search", "knowledge"}),
            source=ToolSource.LOCAL,
            risk_level=ToolRiskLevel.LOW,
            description="Search enterprise knowledge base",
        ),
    )

    # 创建mcp会话
    async with mcp_session() as session:
        # 得到原始mcp tools
        raw_mcp_tools = await get_mcp_tools(session)
        # 转换为langchain可用的tools
        mcp_tools = convert_mcp_tools(session, raw_mcp_tools)

        for tool in mcp_tools:
            registry.register(
                tool,
                ToolMetadata(
                    name=tool.name,
                    category="development",
                    permissions=frozenset({"developer", "admin"}),
                    tags=frozenset({"mcp", "github", "development"}),
                    source=ToolSource.MCP,
                    risk_level=ToolRiskLevel.MEDIUM,
                    description=tool.description or "",
                ),
            )

        yield registry



