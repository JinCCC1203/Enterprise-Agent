from __future__ import annotations

from mcp.server import MCPServer

from .notification import register_notification_tools
from .service import register_service_tools
from .ticket import register_ticket_tools
from .web_search import register_web_search_tools


def register_all_tools(mcp: MCPServer) -> None:
    """
    注册 Enterprise-Agent MCP Server 的全部 Tools。
    """
    register_service_tools(mcp)
    register_ticket_tools(mcp)
    register_notification_tools(mcp)
    register_web_search_tools(mcp)