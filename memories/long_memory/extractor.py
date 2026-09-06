#让 LLM 输出结构化的 MemoryCandidate
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from .models import MemoryType


class MemoryCandidate(BaseModel):
    """
    LLM 提取出的候选长期记忆。
    注意：
    MemoryCandidate 只是“候选记忆”，
    并不代表已经正式写入数据库。
    """

    should_remember: bool = Field(
        description=(
            "Whether this information should be "
            "stored as long-term memory."
        )
    )

    memory_type: MemoryType | None = Field(
        default=None,
        description=(
            "Type of memory: fact, preference, "
            "task, or summary."
        )
    )

    content: str = Field(
        default="",
        description=(
            "A concise, self-contained description "
            "of the information worth remembering."
        )
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence that this information should "
            "be stored as long-term memory."
        )
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional additional metadata about "
            "the memory."
        )
    )


class MemoryExtractor:
    """
    Long-term Memory Extractor。

    职责：
    1. 分析对话
    2. 判断是否存在值得长期保存的信息
    3. 判断记忆类型
    4. 提取标准化 MemoryCandidate
    """

    def __init__(
        self,
        model: ChatOpenAI,
    ) -> None:
        self.model = model

        self.structured_model = (
            self.model.with_structured_output(
                MemoryCandidate
            )
        )

    async def extract(
        self,
        *,
        conversation: str,
    ) -> MemoryCandidate:
        """
        从一段对话中提取候选长期记忆。
        """

        if not conversation.strip():
            return MemoryCandidate(
                should_remember=False,
                memory_type=None,
                content="",
                confidence=0.0,
            )

        prompt = self._build_prompt(
            conversation
        )

        candidate = await self.structured_model.ainvoke(
            prompt
        )

        return self._validate_candidate(
            candidate
        )

    @staticmethod
    def _build_prompt(
        conversation: str,
    ) -> str:
        return f"""
You are a long-term memory extraction component
for an enterprise AI Agent.

Determine whether the conversation contains
information worth remembering across future
conversations.

Only extract information that is:
- useful in future interactions
- reasonably stable
- relevant to the user or their ongoing work

Memory types:

1. FACT
   Stable factual information about the user,
   their work, projects, environment, etc.

2. PREFERENCE
   Long-term preferences such as language,
   answer style, coding preferences, etc.

3. TASK
   Important ongoing task information likely
   to remain useful in future interactions.

4. SUMMARY
   A concise summary of important historical
   context.

Do NOT store:
- greetings
- casual conversation
- temporary information
- irrelevant information
- passwords
- API keys
- tokens
- credentials

If there is no useful long-term memory,
set should_remember=false.

The memory content must be concise,
self-contained, and understandable without
the original conversation.

Conversation:

{conversation}
"""

    @staticmethod
    def _validate_candidate(
        candidate: MemoryCandidate,
    ) -> MemoryCandidate:
        """
        对 LLM 输出进行业务层校验。
        """

        if not candidate.should_remember:
            return MemoryCandidate(
                should_remember=False,
                memory_type=None,
                content="",
                confidence=0.0,
                metadata=candidate.metadata,
            )

        if candidate.memory_type is None:
            raise ValueError(
                "memory_type is required when "
                "should_remember=True"
            )

        if not candidate.content.strip():
            raise ValueError(
                "content is required when "
                "should_remember=True"
            )

        return candidate.model_copy(
            update={
                "content": candidate.content.strip(),
            }
        )