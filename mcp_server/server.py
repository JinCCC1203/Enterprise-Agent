from mcp.server import MCPServer

from tools import register_all_tools

# 创建 MCP Server
mcp = MCPServer("JinC7-agent_server")

# 注册所有 MCP Tools
register_all_tools(mcp)


if __name__ == "__main__":
    mcp.run()