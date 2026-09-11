from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI

from .models import MemoryType


class MemoryCandidate(BaseModel):
    """
    LLM 提取出的候选长期记忆。

    注意：
        MemoryCandidate 只是候选记忆，
        不代表已经正式写入数据库。
    """

    should_remember: bool = Field(
        default=False,
        description=(
            "Whether this information should be "
            "stored as long-term memory."
        ),
    )

    memory_type: MemoryType | None = Field(
        default=None,
        description=(
            "Type of memory: fact, preference, "
            "task, or summary."
        ),
    )

    content: str = Field(
        default="",
        description=(
            "A concise, self-contained description "
            "of the information worth remembering."
        ),
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence that this information should "
            "be stored as long-term memory."
        ),
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional metadata about the memory."
        ),
    )


class MemoryExtractor:
    """
    Long-term Memory Extractor。

    流程：

        Conversation
            ↓
        DeepSeek JSON Output
            ↓
        json.loads()
            ↓
        Pydantic Validation
            ↓
        Business Validation
            ↓
        MemoryCandidate
    """

    def __init__(
        self,
        model: ChatOpenAI,
    ) -> None:
        self.model = model

        # DeepSeek JSON Output。
        #
        # 不使用 with_structured_output，
        # 直接要求模型返回合法 JSON。
        self.json_model = self.model.bind(
            response_format={
                "type": "json_object",
            }
        )

    async def extract(
        self,
        *,
        conversation: str,
    ) -> MemoryCandidate:
        """
        从一段对话中异步提取候选长期记忆。
        """

        if not conversation.strip():
            return MemoryCandidate(
                should_remember=False,
                memory_type=None,
                content="",
                confidence=0.0,
            )

        prompt = self._build_prompt(
            conversation=conversation,
        )

        response = await self.json_model.ainvoke(
            prompt
        )

        content = self._extract_content(
            response.content
        )

        candidate = self._parse_json(
            content
        )

        return self._validate_candidate(
            candidate
        )

    @staticmethod
    def _build_prompt(
        *,
        conversation: str,
    ) -> str:
        """
        构造要求 DeepSeek 返回 JSON 的 Prompt。
        """

        return f"""
You are a long-term memory extraction component
for an enterprise AI Agent.

Analyze the conversation and determine whether it
contains information worth remembering across
future conversations.

You MUST return a valid JSON object only.

Do not return Markdown.
Do not return code fences.
Do not return explanations outside the JSON object.

The JSON object MUST have exactly these fields:

{{
  "should_remember": true,
  "memory_type": "preference",
  "content": "The user prefers responses in Chinese.",
  "confidence": 0.95,
  "metadata": {{}}
}}

Allowed memory_type values:

- "fact"
- "preference"
- "task"
- "summary"

Set "memory_type" to null and "content" to ""
when "should_remember" is false.

Memory types:

1. FACT
   Stable factual information about the user,
   their work, projects, environment, or setup.

2. PREFERENCE
   Long-term preferences such as language,
   answer style, coding preferences, etc.

3. TASK
   Important ongoing task information likely
   to remain useful in future interactions.

4. SUMMARY
   Concise historical context that may be useful
   across future conversations.

Only extract information that is:

- useful in future interactions
- reasonably stable
- relevant to the user or ongoing work
- supported by the conversation

Do NOT store:

- greetings
- casual conversation
- temporary information
- irrelevant information
- passwords
- API keys
- tokens
- credentials
- secret values
- one-time transient values

The memory content must be:

- concise
- self-contained
- factual
- understandable without the original conversation

Conversation:

{conversation}
"""

    @staticmethod
    def _extract_content(
        content: Any,
    ) -> str:
        """
        将 LangChain ModelResponse 的 content
        统一转换成字符串。
        """

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []

            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue

                if isinstance(item, dict):
                    text = item.get("text")

                    if isinstance(text, str):
                        text_parts.append(text)

            return "".join(text_parts).strip()

        raise ValueError(
            "Unexpected model response content type: "
            f"{type(content).__name__}"
        )

    @staticmethod
    def _parse_json(
        content: str,
    ) -> MemoryCandidate:
        """
        将 JSON 字符串转换为 MemoryCandidate。
        """

        if not content:
            raise ValueError(
                "Memory extractor returned empty content."
            )

        # --------------------------------------------------------------
        # 去除可能出现的 Markdown code fence。
        #
        # 虽然 Prompt 明确禁止，
        # 这里仍然做一层容错。
        # --------------------------------------------------------------

        cleaned = content.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        # --------------------------------------------------------------
        # JSON Parse
        # --------------------------------------------------------------

        try:
            data = json.loads(cleaned)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Memory extractor returned invalid JSON: "
                f"{exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Memory extractor JSON output must be an object."
            )

        # --------------------------------------------------------------
        # Pydantic Validation
        # --------------------------------------------------------------

        try:
            return MemoryCandidate.model_validate(
                data
            )

        except ValidationError as exc:
            raise ValueError(
                "Memory candidate validation failed: "
                f"{exc}"
            ) from exc

    @staticmethod
    def _validate_candidate(
        candidate: MemoryCandidate,
    ) -> MemoryCandidate:
        """
        对 Pydantic 校验后的结果进行业务层校验。
        """

        # --------------------------------------------------------------
        # 不需要记忆
        # --------------------------------------------------------------

        if not candidate.should_remember:
            return MemoryCandidate(
                should_remember=False,
                memory_type=None,
                content="",
                confidence=0.0,
                metadata=candidate.metadata,
            )

        # --------------------------------------------------------------
        # should_remember=True 时必须存在 memory_type
        # --------------------------------------------------------------

        if candidate.memory_type is None:
            raise ValueError(
                "memory_type is required when "
                "should_remember=True"
            )

        # --------------------------------------------------------------
        # should_remember=True 时必须存在 content
        # --------------------------------------------------------------

        content = candidate.content.strip()

        if not content:
            raise ValueError(
                "content is required when "
                "should_remember=True"
            )

        # --------------------------------------------------------------
        # confidence 已经由 Pydantic 保证：
        #
        # 0.0 <= confidence <= 1.0
        #
        # 这里进一步返回规范化后的 Candidate。
        # --------------------------------------------------------------

        return candidate.model_copy(
            update={
                "content": content,
            }
        )