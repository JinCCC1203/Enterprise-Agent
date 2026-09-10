from agents.single_agent import create_agent_app
from memories.short_memory import get_config
import asyncio

async def main():
    config=get_config("user_001")
    async with create_agent_app() as agent:
        response = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "请介绍一下ZX Bank的报销制度"
                    }
                ]
            },
            config=config
        )

        print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
