from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage, SystemMessage

from memories.long_memory.manager import MemoryManager
from memories.long_memory.models import MemoryType


class MemoryRetrievalMiddleware(AgentMiddleware):
    """
    在每次 Model Call 前检索长期记忆，
    并将相关 Memory 注入当前 Model Context。
    流程：
        ModelRequest
             ↓
        当前用户 Query
             ↓
        MemoryManager.search()
             ↓
        Relevant Memories
             ↓
        system_message
             ↓
        LLM
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        *,
        top_k: int = 5,
        memory_type: MemoryType | None = None,
        max_memories: int | None = None,
    ) -> None:
        super().__init__()

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if max_memories is not None and max_memories <= 0:
            raise ValueError(
                "max_memories must be greater than 0"
            )

        self.memory_manager = memory_manager
        self.top_k = top_k
        self.memory_type = memory_type
        self.max_memories = max_memories

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[
            [ModelRequest],
            Awaitable[ModelResponse],
        ],
    ) -> ModelResponse:
        """
        Model Call 前检索相关长期记忆。
        """
        # 获取 Runtime Context
        runtime = request.runtime

        if runtime is None:
            return await handler(request)

        context = runtime.context

        # 获取 user_id
        user_id = getattr(
            context,
            "user_id",
            None,
        )

        if not user_id:
            # 没有 user_id 就无法进行用户级 Memory 隔离
            return await handler(request)

        # 提取当前 Query
        query = self._extract_latest_user_query(
            request
        )

        if not query:
            return await handler(request)

        # Semantic Memory Search
        memories = await self.memory_manager.search(
            user_id=user_id,
            query=query,
            top_k=self.top_k,
            memory_type=self.memory_type,
        )

        if not memories:
            return await handler(request)

        # 限制注入 Memory 数量
        if self.max_memories is not None:
            memories = memories[:self.max_memories]

        # 构造 Memory Context
        memory_context = self._build_memory_context(
            memories
        )

        # 注入 System Message
        system_message = request.system_message

        if system_message is None:
            new_system_message = SystemMessage(
                content=memory_context
            )
        else:
            new_system_message = SystemMessage(
                content=(
                    f"{system_message.content}\n\n"
                    f"{memory_context}"
                )
            )

        new_request = request.override(
            system_message=new_system_message,
        )

        return await handler(new_request)

    @staticmethod
    def _extract_latest_user_query(
        request: ModelRequest,
    ) -> str:
        """
        从当前消息中找到最后一条用户消息。
        """

        for message in reversed(request.messages):

            if isinstance(message, HumanMessage):
                content = message.content

                if isinstance(content, str):
                    return content.strip()

        return ""

    @staticmethod
    def _build_memory_context(
        memories,
    ) -> str:
        """
        将 MemorySearchResult 转换成
        可以注入 Model Context 的文本。
        """

        lines = [
            "[Relevant Long-term Memories]",
            (
                "The following memories were retrieved "
                "from the user's long-term memory. "
                "Use them only when relevant to the "
                "current request."
            ),
            "",
        ]

        for index, result in enumerate(
            memories,
            start=1,
        ):
            memory = result.memory

            lines.append(
                f"{index}. "
                f"[{memory.memory_type.value}] "
                f"{memory.content}"
            )

        return "\n".join(lines)


class MemoryPersistenceMiddleware(AgentMiddleware):
    """
    在一次 Agent Run 完成后，
    从本次对话中提取并持久化长期记忆。
    流程：
        Agent Run
             ↓
        aafter_agent
             ↓
        当前 Conversation
             ↓
        MemoryManager.remember()
             ↓
        ADD / UPDATE / IGNORE
             ↓
        PostgreSQL

    不负责：
    - Memory Extraction 的具体实现
    - Embedding
    - PostgreSQL 操作
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
    ) -> None:
        super().__init__()

        self.memory_manager = memory_manager

    async def aafter_agent(
        self,
        state: dict[str, Any],
        runtime: Any,
    ) -> None:
        """
        Agent Run 完成后提取长期记忆。
        """
        # 获取 Runtime Context
        context = runtime.context

        user_id = getattr(
            context,
            "user_id",
            None,
        )
        if not user_id:
            return

        # 获取完整消息历史
        messages = state.get(
            "messages",
            [],
        )

        if not messages:
            return

        # 转换成 MemoryExtractor 可以理解的文本

        conversation = self._build_conversation(
            messages
        )

        if not conversation:
            return

        # 提取并持久化
        await self.memory_manager.remember(
            user_id=user_id,
            conversation=conversation,
        )

    @staticmethod
    def _build_conversation(
        messages: list[Any],
    ) -> str:
        """
        将 Agent 消息历史转换成文本。
        只提取：
        - Human
        - AI
        ToolMessage 暂时不直接纳入 Memory Extraction。
        后续如果需要从 Tool Result 中产生长期记忆，
        再单独增加对应策略。
        """

        conversation_parts: list[str] = []

        for message in messages:

            message_type = getattr(
                message,
                "type",
                None,
            )

            content = getattr(
                message,
                "content",
                "",
            )

            if not isinstance(content, str):
                continue

            content = content.strip()

            if not content:
                continue

            if message_type == "human":
                conversation_parts.append(
                    f"User: {content}"
                )

            elif message_type == "ai":
                conversation_parts.append(
                    f"Assistant: {content}"
                )

        return "\n".join(
            conversation_parts
        )