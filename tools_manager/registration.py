from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp_adapter.mcp_to_langchain import convert_mcp_tools
from mcp_client.client import get_mcp_tools, mcp_session
from tools.rag_tool import (
    RAG_SEARCH_DESCRIPTION,
    rag_search,
)
from tools_manager.metadata import (
    ToolMetadata,
    ToolRiskLevel,
    ToolSource,
)
from tools_manager.registry import ToolRegistry


# ==============================================================
# MCP Tool Metadata
# ==============================================================

MCP_TOOL_METADATA: dict[str, ToolMetadata] = {

    # ==========================================================
    # get_service_health
    # ==========================================================

    "get_service_health": ToolMetadata(
        name="get_service_health",
        category="operations",
        permissions=frozenset(
            {
                "employee",
                "developer",
                "admin",
            }
        ),
        tags=frozenset(
            {
                "mcp",
                "operations",
                "service",
                "health",
                "monitoring",
                "read",
            }
        ),
        source=ToolSource.MCP,
        risk_level=ToolRiskLevel.LOW,
        description=(
            "Query the current health status of an "
            "enterprise service."
        ),
    ),

    # ==========================================================
    # get_ticket
    # ==========================================================

    "get_ticket": ToolMetadata(
        name="get_ticket",
        category="ticketing",
        permissions=frozenset(
            {
                "employee",
                "developer",
                "admin",
            }
        ),
        tags=frozenset(
            {
                "mcp",
                "ticket",
                "incident",
                "read",
                "status",
            }
        ),
        source=ToolSource.MCP,
        risk_level=ToolRiskLevel.LOW,
        description=(
            "Get the current status and details of an "
            "enterprise ticket or incident."
        ),
    ),

    # ==========================================================
    # create_ticket
    # ==========================================================

    "create_ticket": ToolMetadata(
        name="create_ticket",
        category="ticketing",
        permissions=frozenset(
            {
                "developer",
                "admin",
            }
        ),
        tags=frozenset(
            {
                "mcp",
                "ticket",
                "incident",
                "create",
                "write",
            }
        ),
        source=ToolSource.MCP,
        risk_level=ToolRiskLevel.MEDIUM,
        description=(
            "Create a new enterprise incident or ticket. "
            "This is a write operation with an external side "
            "effect and is subject to permission, risk, retry, "
            "and Human-in-the-Loop governance."
        ),
    ),

    # ==========================================================
    # update_ticket
    # ==========================================================

    "update_ticket": ToolMetadata(
        name="update_ticket",
        category="ticketing",
        permissions=frozenset(
            {
                "developer",
                "admin",
            }
        ),
        tags=frozenset(
            {
                "mcp",
                "ticket",
                "incident",
                "update",
                "write",
                "idempotent",
            }
        ),
        source=ToolSource.MCP,
        risk_level=ToolRiskLevel.MEDIUM,
        description=(
            "Update the state or processing comment of an "
            "enterprise ticket or incident. The operation is "
            "designed to be idempotent."
        ),
    ),

    # ==========================================================
    # send_notification
    # ==========================================================

    "send_notification": ToolMetadata(
        name="send_notification",
        category="communication",
        permissions=frozenset(
            {
                "developer",
                "admin",
            }
        ),
        tags=frozenset(
            {
                "mcp",
                "notification",
                "communication",
                "external-effect",
                "write",
            }
        ),
        source=ToolSource.MCP,
        risk_level=ToolRiskLevel.HIGH,
        description=(
            "Send a notification to an enterprise "
            "communication channel. This is a high-risk "
            "external side-effect operation and requires "
            "appropriate risk control and Human-in-the-Loop "
            "approval."
        ),
    ),

    # ==========================================================
    # web_search
    # ==========================================================

    "web_search": ToolMetadata(
        name="web_search",
        category="research",
        permissions=frozenset(
            {
                "employee",
                "developer",
                "admin",
            }
        ),
        tags=frozenset(
            {
                "mcp",
                "research",
                "web",
                "search",
                "external",
                "read",
                "internet",
            }
        ),
        source=ToolSource.MCP,
        risk_level=ToolRiskLevel.LOW,
        description=(
            "Search publicly available information on the "
            "internet, including public documentation, news, "
            "and current external information."
        ),
    ),
}


# ==============================================================
# MCP Metadata Resolver
# ==============================================================

def _metadata_for_mcp_tool(
    tool,
) -> ToolMetadata:
    """
    根据 MCP Tool 名称获取治理 Metadata。

    对未知 MCP Tool 采用保守策略：

    - category = external
    - 仅允许 developer/admin
    - risk = HIGH
    """

    metadata = MCP_TOOL_METADATA.get(
        tool.name
    )

    if metadata is not None:
        return metadata

    return ToolMetadata(
        name=tool.name,
        category="external",
        permissions=frozenset(
            {
                "developer",
                "admin",
            }
        ),
        tags=frozenset(
            {
                "mcp",
                "external",
                "unclassified",
            }
        ),
        source=ToolSource.MCP,
        risk_level=ToolRiskLevel.HIGH,
        description=(
            tool.description
            or "Unclassified external MCP tool."
        ),
    )


# ==============================================================
# Create Tool Registry
# ==============================================================

@asynccontextmanager
async def create_tool_registry(
) -> AsyncIterator[ToolRegistry]:
    """
    创建并维护 Agent Runtime 的统一 ToolRegistry。

    Registry 统一管理：

        1. Local Tools
        2. MCP Tools

    MCP Tool 进入 Registry 前经过：

        MCP Server
            ↓
        MCP Client
            ↓
        list_tools()
            ↓
        MCP Adapter
            ↓
        LangChain StructuredTool
            ↓
        ToolRegistry
            ↓
        ToolMetadata
    """

    registry = ToolRegistry()

    # ==========================================================
    # 1. Local RAG Tool
    # ==========================================================

    registry.register(
        rag_search,
        ToolMetadata(
            name=rag_search.name,
            category="knowledge",
            permissions=frozenset(
                {
                    "employee",
                    "developer",
                    "admin",
                }
            ),
            tags=frozenset(
                {
                    "rag",
                    "search",
                    "knowledge",
                    "enterprise",
                    "internal",
                    "local",
                }
            ),
            source=ToolSource.LOCAL,
            risk_level=ToolRiskLevel.LOW,
            description=(
                RAG_SEARCH_DESCRIPTION
            ),
        ),
    )

    # ==========================================================
    # 2. MCP Session
    # ==========================================================

    async with mcp_session() as session:

        # ------------------------------------------------------
        # 获取 MCP Server 原始 Tools
        # ------------------------------------------------------

        raw_mcp_tools = await get_mcp_tools(
            session
        )

        # ------------------------------------------------------
        # MCP → LangChain Tool
        # ------------------------------------------------------

        mcp_tools = convert_mcp_tools(
            session,
            raw_mcp_tools,
        )

        # ======================================================
        # 3. Register MCP Tools
        # ======================================================

        for tool in mcp_tools:

            metadata = _metadata_for_mcp_tool(
                tool
            )

            registry.register(
                tool,
                metadata,
            )

        # ======================================================
        # 4. Yield Registry
        # ======================================================

        yield registry