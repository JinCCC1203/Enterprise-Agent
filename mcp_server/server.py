from mcp.server import MCPServer
from tools.knowledge import register_knowledge_tools

#创建mcp服务器
mcp=MCPServer("JinC7-agent_server")
#注册knowledge Tool
register_knowledge_tools(mcp)

if __name__ == "__main__":
    mcp.run()