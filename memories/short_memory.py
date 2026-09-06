from langgraph.checkpoint.memory import InMemorySaver

checkpointer=InMemorySaver()


def get_config(thread_id: str) -> dict:
    """根据会话 ID 创建 Agent 配置"""
    return {
        "configurable":{
            "thread_id":thread_id
        }
    }
