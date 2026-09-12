from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.runtime import Runtime

from memories.long_memory.manager import MemoryManager
from memories.long_memory.models import MemoryType
from workflow.state import (
    EnterpriseAgentContext,
    EnterpriseAgentState,
)
from langchain_core.runnables import RunnableConfig

def create_memory_retrieval_node(
    *,
    memory_manager: MemoryManager,
    top_k: int = 5,
    memory_type: MemoryType | None = None,
):
    """
    创建 LangGraph Memory Retrieval Node。

    Workflow 中的职责：

        User Query
            ↓
        Runtime Context.user_id
            ↓
        MemoryManager.search()
            ↓
        Relevant Long-term Memories
            ↓
        EnterpriseAgentState["retrieved_memories"]

    设计原则：
    1. Node 只负责 Workflow 编排。
    2. 不直接操作 MemoryStore。
    3. 不直接生成 embedding。
    4. user_id 来自可信 Runtime Context，而不是 State / Memory。
    5. 检索结果写入 Graph State，后续 Agent 共享。
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )

    async def memory_retrieval_node(
        state: EnterpriseAgentState,
        runtime: Runtime[EnterpriseAgentContext],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """
        LangGraph Memory Retrieval Node。

        参数：
            state:
                当前 Graph State。

            runtime:
                LangGraph Runtime。
                user_id 从 runtime.context 获取。

            config:
                当前 Graph execution config。
                当前 Node 不直接使用，但保留该参数，
                方便后续 checkpoint / tracing / recovery。
        """

        # --------------------------------------------------------------
        # 1. 获取 Runtime Context
        # --------------------------------------------------------------

        context = runtime.context

        user_id = context.user_id

        if not user_id:
            raise ValueError(
                "Memory retrieval requires a valid user_id."
            )

        # --------------------------------------------------------------
        # 2. 获取当前用户 Query
        # --------------------------------------------------------------

        messages = state.get(
            "messages",
            [],
        )

        query = _extract_latest_user_query(
            messages
        )

        # 没有用户 Query 时，不进行 Memory Search。
        if not query:
            return {
                "retrieved_memories": [],
            }

        # --------------------------------------------------------------
        # 3. Semantic Memory Search
        # --------------------------------------------------------------

        results = await memory_manager.search(
            user_id=user_id,
            query=query,
            top_k=top_k,
            memory_type=memory_type,
        )

        # --------------------------------------------------------------
        # 4. Convert MemorySearchResult
        #    -> Graph State
        #
        # State 当前设计为 list[str]，
        # 因此这里只保存适合后续 Agent 使用的文本。
        #
        # distance 不直接暴露给 Agent。
        # MemoryManager 已经负责 relevance threshold。
        # --------------------------------------------------------------

        retrieved_memories = [
            _format_memory_result(result)
            for result in results
        ]

        # --------------------------------------------------------------
        # 5. 写入 Graph State
        # --------------------------------------------------------------

        return {
            "retrieved_memories": retrieved_memories,
        }

    return memory_retrieval_node


def _extract_latest_user_query(
    messages: list[BaseMessage],
) -> str:
    """
    从当前 Workflow Message State 中提取
    最后一条 Human Message。

    Retrieval Node 使用“当前用户问题”作为
    semantic memory search query，而不是整个
    conversation。
    """

    for message in reversed(messages):
        if not isinstance(
            message,
            HumanMessage,
        ):
            continue

        content = message.content

        if isinstance(content, str):
            content = content.strip()

            if content:
                return content

        # ----------------------------------------------------------
        # 兼容部分 LangChain message content block 格式：
        #
        # [
        #     {"type": "text", "text": "..."},
        #     ...
        # ]
        # ----------------------------------------------------------

        if isinstance(content, list):
            text_parts: list[str] = []

            for block in content:
                if isinstance(
                    block,
                    str,
                ):
                    text_parts.append(block)
                    continue

                if not isinstance(
                    block,
                    dict,
                ):
                    continue

                text = block.get("text")

                if isinstance(
                    text,
                    str,
                ):
                    text_parts.append(text)

            query = "".join(
                text_parts
            ).strip()

            if query:
                return query

    return ""


def _format_memory_result(
    result: Any,
) -> str:
    """
    将 MemorySearchResult 转换成
    可以安全共享给 Supervisor / Specialist Agent
    的文本。

    示例：

        [preference] 用户偏好使用中文回答

    不直接向 LLM 暴露数据库 ID 或 embedding。
    """

    memory = result.memory

    memory_type = (
        memory.memory_type.value
        if memory.memory_type is not None
        else "unknown"
    )

    content = memory.content.strip()

    return (
        f"[{memory_type}] "
        f"{content}"
    )

