from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from mcp import ClientSession
from mcp.client.stdio import stdio_client,StdioServerParameters

server_params=StdioServerParameters(
    command="python",
    args=["mcp_server/server"]
)

@asynccontextmanager
async def mcp_session() -> AsyncIterator[ClientSession]:
    """
    创建并维护一个 MCP ClientSession

    这个会话将在整个 async with 生命周期内保持有效
    """
    async with stdio_client(server_params) as streams:
        read_stream,write_stream=streams
        async with ClientSession(read_stream,write_stream) as session:
            #mcp初始化握手
            await session.initialize()
            print("已经初始化 MCP 会话")
            #将已经初始化的 session 将给 Agent
            yield session
            print("MCP 会话关闭")

async def get_mcp_tools(
        session: ClientSession
) -> list:
    """获取 MCP server 暴露的 tools"""

    #列出可用工具
    response=await session.list_tools()
    print(f"可用的 MCP tools:")
    for tool in response.tools:
        print(f"工具名称: {tool.name}")
        print(f"工具描述: {tool.description}")
        print(f"工具输入参数: {tool.input_schema}")
        print("="*20)

    return response.tools

