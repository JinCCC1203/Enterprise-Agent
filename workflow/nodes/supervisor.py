from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field, ValidationError

from workflow.state import (
    EnterpriseAgentContext,
    EnterpriseAgentState,
)


SupervisorAgent = Literal[
    "knowledge_agent",
    "operations_agent",
    "ticket_agent",
    "research_agent",
    "end",
]


class SupervisorDecision(BaseModel):
    """
    Supervisor LLM 的结构化决策结果。

    next_agent:
        决定下一步进入哪个 Specialist Agent。

    reason:
        说明为什么选择该 Agent。

    task:
        对当前任务进行简要描述，
        供后续 Specialist Agent 理解 Supervisor 的决策。
    """

    next_agent: SupervisorAgent = Field(
        description=(
            "The next agent to execute. "
            "Must be one of: "
            "knowledge_agent, operations_agent, "
            "ticket_agent, notification_agent, end."
        )
    )

    reason: str = Field(
        min_length=1,
        description="Reason for routing the task.",
    )

    task: str = Field(
        min_length=1,
        description="Concise task description for the next agent.",
    )


def create_supervisor_node(
    *,
    model: ChatOpenAI,
):
    """
    创建 Supervisor Node。

    Supervisor 的职责：

        当前用户请求
              +
        Retrieved Memories
              +
        当前 Workflow State
              ↓
        LLM 判断任务应该交给哪个 Specialist
              ↓
        SupervisorDecision
              ↓
        Graph State["next_agent"]

    注意：

    Supervisor 不直接执行 Tool。
    Supervisor 也不负责最终的 Graph edge routing。
    它只负责产生路由决策。
    """

    json_model = model.bind(
        response_format={
            "type": "json_object",
        }
    )

    async def supervisor_node(
        state: EnterpriseAgentState,
        runtime: Runtime[EnterpriseAgentContext],
        config: Any,
    ) -> dict[str, Any]:
        """
        LangGraph Supervisor Node。
        """

        messages = state.get(
            "messages",
            [],
        )

        retrieved_memories = state.get(
            "retrieved_memories",
            [],
        )

        # ----------------------------------------------------------
        # 1. 获取当前用户请求
        # ----------------------------------------------------------

        user_query = _extract_latest_user_query(
            messages
        )

        if not user_query:
            return {
                "next_agent": "end",
                "task_status": "completed",
                "handoff_reason": (
                    "No user query was found."
                ),
            }

        # ----------------------------------------------------------
        # 2. 构造 Supervisor Prompt
        # ----------------------------------------------------------

        prompt = _build_supervisor_prompt(
            user_query=user_query,
            retrieved_memories=retrieved_memories,
            state=state,
        )

        # ----------------------------------------------------------
        # 3. 调用 LLM
        # ----------------------------------------------------------

        response = await json_model.ainvoke(
            prompt,
            config=config,
        )

        # ----------------------------------------------------------
        # 4. 提取模型输出
        # ----------------------------------------------------------

        raw_content = _extract_response_content(
            response.content
        )

        # ----------------------------------------------------------
        # 5. JSON Parse + Pydantic Validation
        # ----------------------------------------------------------

        decision = _parse_decision(
            raw_content
        )

        # ----------------------------------------------------------
        # 6. 写入 Graph State
        # ----------------------------------------------------------

        return {
            "next_agent": decision.next_agent,
            "handoff_reason": decision.reason,
            "task_status": (
                "completed"
                if decision.next_agent == "end"
                else "running"
            ),
        }

    return supervisor_node


def _build_supervisor_prompt(
    *,
    user_query: str,
    retrieved_memories: list[str],
    state: EnterpriseAgentState,
) -> str:
    """
    构建 Supervisor Prompt。

    Supervisor 只负责：
        task understanding
        agent selection
        routing reason

    不负责工具调用。
    """

    memories_text = (
        "\n".join(
            f"- {memory}"
            for memory in retrieved_memories
        )
        if retrieved_memories
        else "No relevant long-term memories."
    )

    current_agent = state.get(
        "current_agent"
    )

    return f"""
You are the Supervisor of an enterprise multi-agent system.

Your responsibility is to decide which specialist agent
should handle the user's current request.

You MUST NOT execute tools.
You MUST NOT answer the user directly.
You MUST ONLY return a routing decision.

Available specialist agents:

1. knowledge_agent
   - Enterprise knowledge retrieval
   - RAG search
   - Documents, policies, internal knowledge

2. operations_agent
   - Service health
   - System status
   - Operational diagnostics
   - Read-only infrastructure operations

3. ticket_agent
   - Query tickets
   - Create tickets
   - Update tickets
   - Ticket lifecycle operations

4. notification_agent
   - Send notifications
   - External communication
   - Potentially high-risk side effects

5. end
   - Use when the task is already complete,
     unnecessary to call another specialist,
     or the request cannot be meaningfully routed.

Current user query:
{user_query}

Relevant long-term memories:
{memories_text}

Current agent:
{current_agent or "none"}

Current task status:
{state.get("task_status", "unknown")}

Return ONLY valid JSON.

Required format:

{{
  "next_agent": "knowledge_agent | operations_agent | ticket_agent | notification_agent | end",
  "reason": "Why this agent should handle the task",
  "task": "Concise description of the task for the next agent"
}}
""".strip()


def _extract_latest_user_query(
    messages: list[BaseMessage],
) -> str:
    """
    从 Graph Message State 中取得最后一条用户消息。
    """

    for message in reversed(messages):

        message_type = getattr(
            message,
            "type",
            None,
        )

        if message_type != "human":
            continue

        content = message.content

        if isinstance(
            content,
            str,
        ):
            content = content.strip()

            if content:
                return content

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
                    text_parts.append(block)
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
                    text_parts.append(text)

            query = "".join(
                text_parts
            ).strip()

            if query:
                return query

    return ""


def _extract_response_content(
    content: Any,
) -> str:
    """
    兼容不同 ChatModel response.content 格式。
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
                text_parts.append(block)
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
                text_parts.append(text)

        return "".join(
            text_parts
        ).strip()

    raise ValueError(
        "Supervisor model returned unsupported content type."
    )


def _parse_decision(
    raw_content: str,
) -> SupervisorDecision:
    """
    JSON Parse + Pydantic validation。
    """

    if not raw_content:
        raise ValueError(
            "Supervisor returned empty content."
        )

    cleaned = raw_content.strip()

    # 防御性处理 Markdown JSON Fence
    if cleaned.startswith(
        "```"
    ):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(
            lines
        ).strip()

    try:
        payload = json.loads(
            cleaned
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Supervisor returned invalid JSON."
        ) from exc

    try:
        decision = SupervisorDecision.model_validate(
            payload
        )

    except ValidationError as exc:
        raise ValueError(
            f"Invalid Supervisor decision: {exc}"
        ) from exc

    return decision