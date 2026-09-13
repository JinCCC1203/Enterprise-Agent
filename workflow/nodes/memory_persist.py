from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.runtime import Runtime

from memories.long_memory.manager import (
    MemoryManager,
    MemoryOperation,
)
from workflow.state import (
    EnterpriseAgentContext,
    EnterpriseAgentState,
)


def create_memory_persist_node(
    *,
    memory_manager: MemoryManager,
):
    """
    创建 LangGraph Memory Persistence Node。

    Workflow 中的职责：

        Current Workflow
              ↓
        User Query + Agent Response
              ↓
        MemoryManager.remember()
              ↓
        MemoryExtractor
              ↓
        Confidence Filter
              ↓
        Embedding
              ↓
        Duplicate Detection
              ↓
        ADD / UPDATE / IGNORE
              ↓
        PostgreSQL + pgvector

    注意：

        Node 只负责 Workflow 编排。

        MemoryManager 负责：
        - Memory Extraction
        - Confidence Filtering
        - Embedding
        - Duplicate Detection
        - ADD / UPDATE / IGNORE
        - Persistence
    """

    async def memory_persist_node(
        state: EnterpriseAgentState,
        runtime: Runtime[EnterpriseAgentContext],
    ) -> dict[str, Any]:
        """
        LangGraph Memory Persistence Node。

        user_id 必须来自 Runtime Context，
        不从 Graph State 中读取。
        """

        # --------------------------------------------------------------
        # 1. Runtime Identity
        # --------------------------------------------------------------

        context = runtime.context

        user_id = context.user_id

        if not user_id:
            raise ValueError(
                "Memory persistence requires a valid user_id."
            )

        # --------------------------------------------------------------
        # 2. 获取当前 Workflow Messages
        # --------------------------------------------------------------

        messages = state.get(
            "messages",
            [],
        )

        if not messages:
            return {
                "memory_persisted": False,
                "memory_operation": "ignore",
                "memory_persist_reason": (
                    "No workflow messages were available."
                ),
            }

        # --------------------------------------------------------------
        # 3. 提取本次 Workflow 的 Memory Candidate Source
        #
        # 当前版本只使用：
        #
        #     User Query
        #          +
        #     Final Agent Response
        #
        # 不把中间 ToolMessage /
        # Tool execution details 直接送给 MemoryExtractor。
        # --------------------------------------------------------------

        conversation = _build_memory_conversation(
            messages
        )

        if not conversation:
            return {
                "memory_persisted": False,
                "memory_operation": "ignore",
                "memory_persist_reason": (
                    "No suitable user/assistant conversation "
                    "was found."
                ),
            }

        # --------------------------------------------------------------
        # 4. MemoryManager.remember()
        # --------------------------------------------------------------

        try:
            result = await memory_manager.remember(
                user_id=user_id,
                conversation=conversation,
            )

        except Exception as exc:
            # ----------------------------------------------------------
            # Persistence failure should not automatically invalidate
            # the successful Agent Workflow.
            #
            # Store the failure in Graph State so that a later
            # Recovery / Persistence Retry mechanism can handle it.
            # ----------------------------------------------------------

            return {
                "memory_persisted": False,
                "memory_operation": "error",
                "memory_persist_reason": (
                    f"Memory persistence failed: {exc}"
                ),
            }

        # --------------------------------------------------------------
        # 5. Convert MemoryOperationResult
        #    -> Graph State
        # --------------------------------------------------------------

        operation = result.operation.value

        persisted = (
            result.operation
            in {
                MemoryOperation.ADD,
                MemoryOperation.UPDATE,
            }
        )

        return {
            "memory_persisted": persisted,
            "memory_operation": operation,
            "memory_persist_reason": result.reason,
        }

    return memory_persist_node


def _build_memory_conversation(
    messages: list[BaseMessage],
) -> str:
    """
    构造 MemoryExtractor 的输入。

    当前策略：

        最后一条 HumanMessage
                +
        最后一条 AIMessage

    示例：

        User:
        我以后都希望使用中文回答。

        Assistant:
        好的，我之后会优先使用中文回答。

    这样 MemoryExtractor 能够判断：
        - preference
        - fact
        - task
        - summary

    而不会把大量 ToolMessage、
    MCP JSON 输出和中间 Agent 思考过程
    直接写入长期记忆。
    """

    user_message = _extract_latest_message(
        messages=messages,
        message_type="human",
    )

    assistant_message = _extract_latest_message(
        messages=messages,
        message_type="ai",
    )

    if not user_message:
        return ""

    if assistant_message:
        return (
            f"User:\n"
            f"{user_message}\n\n"
            f"Assistant:\n"
            f"{assistant_message}"
        )

    return (
        f"User:\n"
        f"{user_message}"
    )


def _extract_latest_message(
    *,
    messages: list[BaseMessage],
    message_type: str,
) -> str:
    """
    获取最后一条指定类型的消息。

    支持：
        HumanMessage
        AIMessage
    """

    for message in reversed(messages):

        if message_type == "human":
            if not isinstance(
                message,
                HumanMessage,
            ):
                continue

        elif message_type == "ai":
            if not isinstance(
                message,
                AIMessage,
            ):
                continue

        else:
            continue

        content = message.content

        text = _extract_text_content(
            content
        )

        if text:
            return text

    return ""


def _extract_text_content(
    content: Any,
) -> str:
    """
    兼容 LangChain message.content 的常见格式。

    支持：

        str

        [
            {"type": "text", "text": "..."}
        ]

        ["...", "..."]
    """

    if isinstance(
        content,
        str,
    ):
        return content.strip()

    if isinstance(
        content,
        list,
    ):
        text_parts: list[str] = []

        for block in content:

            if isinstance(
                block,
                str,
            ):
                text_parts.append(
                    block
                )
                continue

            if not isinstance(
                block,
                dict,
            ):
                continue

            text = block.get(
                "text"
            )

            if isinstance(
                text,
                str,
            ):
                text_parts.append(
                    text
                )

        return "".join(
            text_parts
        ).strip()

    return ""
