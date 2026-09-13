from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from .models import MemoryType


class MemoryCandidate(BaseModel):
    """
    LLM 提取出的候选长期记忆。

    注意：
        MemoryCandidate 只是“候选记忆”，
        不代表已经正式写入数据库。

    真正的 ADD / UPDATE / IGNORE
    由 MemoryManager 决定。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

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
        min_length=0,
        max_length=1000,
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

    职责：

        Terminal Workflow
            ↓
        Curated Conversation
            ↓
        DeepSeek JSON Output
            ↓
        JSON Parse
            ↓
        Pydantic Validation
            ↓
        Business Validation
            ↓
        MemoryCandidate

    不负责：

        - Embedding
        - Similarity Search
        - Duplicate Detection
        - Conflict Resolution
        - ADD
        - UPDATE
        - IGNORE
        - PostgreSQL Persistence
    """

    MAX_CONTENT_LENGTH = 1000

    def __init__(
        self,
        model: ChatOpenAI,
    ) -> None:
        self.model = model

        # ----------------------------------------------------------
        # DeepSeek JSON Output
        # ----------------------------------------------------------
        #
        # 当前项目已经验证 deepseek-v4-flash 的 JSON Object
        # 调用方式，因此继续使用该方式。
        # ----------------------------------------------------------

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
        从一次 Terminal Workflow 的整理后上下文中，
        提取一个候选长期记忆。

        conversation 应由 Memory Persist Node 构造，
        不建议直接把原始 MCP / Tool 输出全部传入。

        没有值得记忆的内容时：

            should_remember = False
        """

        # ----------------------------------------------------------
        # 1. Empty Input
        # ----------------------------------------------------------

        normalized_conversation = (
            conversation.strip()
        )

        if not normalized_conversation:
            return self._empty_candidate()

        # ----------------------------------------------------------
        # 2. Prompt
        # ----------------------------------------------------------

        prompt = self._build_prompt(
            conversation=normalized_conversation,
        )

        # ----------------------------------------------------------
        # 3. Model
        # ----------------------------------------------------------

        response = await self.json_model.ainvoke(
            prompt
        )

        # ----------------------------------------------------------
        # 4. Normalize Model Content
        # ----------------------------------------------------------

        content = self._extract_content(
            response.content
        )

        # ----------------------------------------------------------
        # 5. Parse JSON
        # ----------------------------------------------------------

        candidate = self._parse_json(
            content
        )

        # ----------------------------------------------------------
        # 6. Business Validation
        # ----------------------------------------------------------

        return self._validate_candidate(
            candidate
        )

    # ==================================================================
    # Prompt
    # ==================================================================

    @classmethod
    def _build_prompt(
        cls,
        *,
        conversation: str,
    ) -> str:
        """
        构造 Memory Extraction Prompt。
        """

        return f"""
You are the long-term memory extraction component
of an enterprise AI Agent Runtime.

Your task is to determine whether the provided
terminal workflow contains information that is
valuable for future interactions with the SAME user.

You MUST return exactly one valid JSON object.

Do NOT return:
- Markdown
- code fences
- explanations
- comments
- text outside the JSON object

The JSON object MUST follow this schema:

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

When "should_remember" is false, return:

{{
  "should_remember": false,
  "memory_type": null,
  "content": "",
  "confidence": 0.0,
  "metadata": {{}}
}}

============================================================
MEMORY TYPES
============================================================

1. FACT

Stable factual information about:

- the user
- the user's work
- the user's projects
- the user's technical environment
- stable enterprise context

Examples:

- The user is working on an enterprise Agent Runtime.
- The user is using PostgreSQL for long-term memory.

Do NOT store information that is only true for the
current single request.

------------------------------------------------------------

2. PREFERENCE

Long-term preferences about how the user wants the
Agent to behave.

Examples:

- The user prefers responses in Chinese.
- The user prefers concise technical explanations.
- The user prefers Python examples.

Only store preferences when they are explicitly
stated or strongly supported by the conversation.

------------------------------------------------------------

3. TASK

Important ongoing work that is likely to remain
useful across future interactions.

Examples:

- The user is currently implementing the Recovery
  layer of an Enterprise Agent Runtime.
- The user is building a multi-agent workflow using
  LangGraph.

Do NOT store one-time requests as tasks.

------------------------------------------------------------

4. SUMMARY

Concise durable historical context that may improve
future assistance.

Use this type sparingly.

============================================================
WHAT SHOULD BE REMEMBERED
============================================================

Only extract information that is:

- useful in future interactions
- reasonably stable
- relevant to the user or their ongoing work
- supported by the provided workflow context
- understandable without the original conversation

The memory should be:

- concise
- self-contained
- factual
- specific
- reusable

============================================================
DO NOT STORE
============================================================

Do NOT store:

- greetings
- casual conversation
- generic questions
- one-time requests
- temporary runtime state
- transient errors
- temporary tool failures
- API responses
- raw MCP output
- raw ToolMessage content
- URLs
- timestamps
- request IDs
- internal execution IDs
- model reasoning
- system prompts
- secrets
- passwords
- API keys
- access tokens
- credentials
- authentication information

Do NOT transform a temporary failure into a
long-term memory.

For example:

BAD:
"The web_search tool timed out."

BAD:
"The user asked about DeepSeek today."

GOOD:
"The user is building an enterprise Agent Runtime
with LangGraph and MCP."

============================================================
MEMORY CONTENT REQUIREMENTS
============================================================

The content must:

1. describe exactly ONE meaningful memory;
2. be self-contained;
3. avoid references such as "today", "this request",
   "above", or "the current conversation";
4. not include secrets or credentials;
5. avoid unnecessary implementation details;
6. be useful without access to the original workflow.

============================================================
CONFIDENCE
============================================================

confidence must be between 0.0 and 1.0.

Use a high confidence only when the information is
explicitly stated or strongly supported.

Examples:

0.95 - explicitly stated stable preference/fact
0.85 - strongly supported by multiple messages
0.70 - reasonably supported but somewhat uncertain
below 0.70 - usually should not be remembered

============================================================
METADATA
============================================================

metadata is optional.

Only include non-sensitive metadata useful for
memory management.

Never include:

- passwords
- API keys
- tokens
- credentials
- authorization headers
- secrets

============================================================
IMPORTANT
============================================================

The workflow may have completed successfully or
failed.

A failed workflow can still contain a valid long-term
memory candidate.

Judge the VALUE and STABILITY of the information,
not merely whether the workflow succeeded.

If there is no valuable long-term memory candidate,
return should_remember=false.

============================================================
TERMINAL WORKFLOW CONTEXT
============================================================

{conversation}
"""

    # ==================================================================
    # Content Normalization
    # ==================================================================

    @staticmethod
    def _extract_content(
        content: Any,
    ) -> str:
        """
        将 LangChain ModelResponse.content
        统一转换为字符串。

        支持：

            str

        或：

            [
                {"type": "text", "text": "..."}
            ]

        或：

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

            for item in content:

                if isinstance(
                    item,
                    str,
                ):
                    text_parts.append(
                        item
                    )
                    continue

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                text = item.get(
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

        raise ValueError(
            "Unexpected model response content type: "
            f"{type(content).__name__}"
        )

    # ==================================================================
    # JSON Parse
    # ==================================================================

    @classmethod
    def _parse_json(
        cls,
        content: str,
    ) -> MemoryCandidate:
        """
        将 JSON 字符串转换为 MemoryCandidate。

        Prompt 要求纯 JSON，
        这里仍然对 Markdown code fence 做容错。
        """

        if not content.strip():
            raise ValueError(
                "Memory extractor returned empty content."
            )

        cleaned = content.strip()

        # ----------------------------------------------------------
        # Remove Markdown code fence if model accidentally adds it.
        # ----------------------------------------------------------

        if cleaned.startswith(
            "```"
        ):

            lines = cleaned.splitlines()

            # 第一行：
            # ```json
            # 或
            # ```
            if lines:
                lines = lines[1:]

            # 最后一行：
            # ```
            if (
                lines
                and lines[-1].strip()
                == "```"
            ):
                lines = lines[:-1]

            cleaned = "\n".join(
                lines
            ).strip()

        # ----------------------------------------------------------
        # JSON Parse
        # ----------------------------------------------------------

        try:

            data = json.loads(
                cleaned
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Memory extractor returned "
                f"invalid JSON: {exc}"
            ) from exc

        # ----------------------------------------------------------
        # Root Type
        # ----------------------------------------------------------

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "Memory extractor JSON output "
                "must be an object."
            )

        # ----------------------------------------------------------
        # Pydantic Validation
        # ----------------------------------------------------------

        try:

            return MemoryCandidate.model_validate(
                data
            )

        except ValidationError as exc:

            raise ValueError(
                "Memory candidate validation failed: "
                f"{exc}"
            ) from exc

    # ==================================================================
    # Business Validation
    # ==================================================================

    @classmethod
    def _validate_candidate(
        cls,
        candidate: MemoryCandidate,
    ) -> MemoryCandidate:
        """
        对 Pydantic 校验后的 Candidate
        进行业务层校验。
        """

        # ----------------------------------------------------------
        # Normalize Content
        # ----------------------------------------------------------

        normalized_content = (
            candidate.content.strip()
        )

        # ----------------------------------------------------------
        # should_remember=False
        #
        # 无论模型返回了什么内容，
        # 最终统一成标准 Empty Candidate。
        # ----------------------------------------------------------

        if not candidate.should_remember:

            return MemoryCandidate(
                should_remember=False,
                memory_type=None,
                content="",
                confidence=0.0,
                metadata=dict(
                    candidate.metadata
                ),
            )

        # ----------------------------------------------------------
        # memory_type required
        # ----------------------------------------------------------

        if candidate.memory_type is None:

            raise ValueError(
                "memory_type is required when "
                "should_remember=True."
            )

        # ----------------------------------------------------------
        # content required
        # ----------------------------------------------------------

        if not normalized_content:

            raise ValueError(
                "content is required when "
                "should_remember=True."
            )

        # ----------------------------------------------------------
        # Content length
        #
        # Pydantic 已经有 max_length，
        # 这里再做一次业务层保护。
        # ----------------------------------------------------------

        if len(
            normalized_content
        ) > cls.MAX_CONTENT_LENGTH:

            raise ValueError(
                "Memory content exceeds the maximum "
                f"length of {cls.MAX_CONTENT_LENGTH}."
            )

        # ----------------------------------------------------------
        # Confidence
        #
        # Pydantic 已经验证：
        #
        #     0.0 <= confidence <= 1.0
        #
        # 这里保留原值，不擅自修改 LLM 的评分。
        # ----------------------------------------------------------

        return candidate.model_copy(
            update={
                "content": normalized_content,
                "metadata": dict(
                    candidate.metadata
                ),
            }
        )

    # ==================================================================
    # Empty Candidate
    # ==================================================================

    @staticmethod
    def _empty_candidate() -> MemoryCandidate:
        """
        返回标准的“不需要记忆” Candidate。
        """

        return MemoryCandidate(
            should_remember=False,
            memory_type=None,
            content="",
            confidence=0.0,
            metadata={},
        )