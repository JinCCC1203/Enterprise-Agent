from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from workflow.state import EnterpriseAgentState


# ==============================================================
# Supervisor Decision
# ==============================================================

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
        决定下一步应该执行哪个 Specialist，
        或者 end 表示整个 Workflow 已完成。

    reason:
        当前路由决策的原因。

    task:
        给下一 Specialist 的具体任务描述。
        该字段会被写入 Graph State 的 handoff_task。

    workflow_complete:
        是否整个 Workflow 已经完成。
    """

    next_agent: SupervisorAgent = Field(
        description=(
            "The next agent to execute. "
            "Must be one of: "
            "knowledge_agent, operations_agent, "
            "ticket_agent, research_agent, end."
        ),
    )

    reason: str = Field(
        min_length=1,
        description=(
            "Reason for the current routing decision."
        ),
    )

    task: str = Field(
        default=(
            "Continue the workflow based on "
            "the selected agent."
        ),
        min_length=1,
        description=(
            "Concise task description for the selected "
            "specialist agent."
        ),
    )

    workflow_complete: bool = Field(
        description=(
            "Whether the entire user workflow is complete. "
            "True only when no further specialist execution "
            "is required."
        ),
    )

    @field_validator(
        "task",
        mode="before",
    )
    @classmethod
    def normalize_task(
        cls,
        value: object,
    ) -> str:
        """
        归一化 LLM 可能返回的空 task。

        某些模型在 JSON structured output 下可能返回：

            "task": ""
            "task": null

        task 当前是 Specialist handoff 的辅助字段，
        不应该因为空字符串直接导致整个 Workflow 失败。
        """

        if value is None:
            return (
                "Continue the workflow based on "
                "the selected agent."
            )

        if not isinstance(
            value,
            str,
        ):
            value = str(value)

        value = value.strip()

        if not value:
            return (
                "Continue the workflow based on "
                "the selected agent."
            )

        return value


# ==============================================================
# Supervisor Node Factory
# ==============================================================

def create_supervisor_node(
    *,
    model: ChatOpenAI,
):
    """
    创建 Supervisor Node。

    Supervisor 是 Enterprise-Agent 的 Workflow Planner。

    每次执行都会综合：

        1. 用户原始 Query
        2. Retrieved Memories
        3. 当前 Workflow State
        4. 已完成 Specialist
        5. Tool Execution Results
        6. 当前 Specialist
        7. 当前 handoff_task
        8. 当前 task_status

    然后决定：

        Specialist A
            ↓
        Supervisor
            ↓
        Specialist B
            ↓
        Supervisor
            ↓
           END

    Supervisor 不直接执行 Tool。
    Supervisor 不生成最终用户答案。
    Supervisor 只负责 Workflow Planning / Routing。
    """

    json_model = model.bind(
        response_format={
            "type": "json_object",
        }
    )

    async def supervisor_node(
        state: EnterpriseAgentState,
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

        completed_agents = state.get(
            "completed_agents",
            [],
        )

        tool_results = state.get(
            "tool_results",
            [],
        )

        current_agent = state.get(
            "current_agent",
        )

        handoff_task = state.get(
            "handoff_task",
        )

        task_status = state.get(
            "task_status",
            "unknown",
        )

        workflow_complete = state.get(
            "workflow_complete",
            False,
        )

        # ----------------------------------------------------------
        # 1. 获取用户 Query
        # ----------------------------------------------------------

        user_query = _extract_latest_user_query(
            messages
        )

        if not user_query:
            return {
                "next_agent": "end",
                "workflow_complete": True,
                "task_status": "completed",
                "handoff_reason": (
                    "No user query was found."
                ),
                "handoff_task": None,
                "current_agent": None,
            }

        # ----------------------------------------------------------
        # 2. 如果 Workflow 已经完成
        # ----------------------------------------------------------

        if workflow_complete:
            return {
                "next_agent": "end",
                "workflow_complete": True,
                "task_status": "completed",
                "handoff_reason": (
                    "The workflow has already been "
                    "completed."
                ),
                "handoff_task": None,
                "current_agent": None,
            }

        # ----------------------------------------------------------
        # 3. 构造 Supervisor Prompt
        # ----------------------------------------------------------

        prompt = _build_supervisor_prompt(
            user_query=user_query,
            retrieved_memories=retrieved_memories,
            completed_agents=completed_agents,
            tool_results=tool_results,
            current_agent=current_agent,
            handoff_task=handoff_task,
            task_status=task_status,
        )

        # ----------------------------------------------------------
        # 4. 调用 LLM
        # ----------------------------------------------------------

        response = await json_model.ainvoke(
            prompt,
        )

        # ----------------------------------------------------------
        # 5. 提取模型输出
        # ----------------------------------------------------------

        raw_content = _extract_response_content(
            response.content
        )

        # ----------------------------------------------------------
        # 6. Parse + Validation
        # ----------------------------------------------------------

        decision = _parse_decision(
            raw_content
        )

        # ----------------------------------------------------------
        # 7. 防御性修正
        #
        # end：
        #   workflow_complete=True
        #
        # specialist：
        #   workflow_complete=False
        # ----------------------------------------------------------

        if decision.next_agent == "end":
            final_workflow_complete = True
            final_task_status = "completed"
            next_current_agent = None
            next_handoff_task = None

        else:
            final_workflow_complete = False
            final_task_status = "running"
            next_current_agent = decision.next_agent
            next_handoff_task = decision.task

        # ----------------------------------------------------------
        # 8. 写入 Graph State
        # ----------------------------------------------------------

        return {
            "next_agent": decision.next_agent,
            "current_agent": next_current_agent,
            "handoff_task": next_handoff_task,
            "workflow_complete": final_workflow_complete,
            "task_status": final_task_status,
            "handoff_reason": decision.reason,
        }

    return supervisor_node


# ==============================================================
# Prompt Construction
# ==============================================================

def _build_supervisor_prompt(
    *,
    user_query: str,
    retrieved_memories: list[str],
    completed_agents: list[str],
    tool_results: list[Any],
    current_agent: str | None,
    handoff_task: str | None,
    task_status: str,
) -> str:
    """
    构造 Multi-Specialist Supervisor Prompt。

    Supervisor 的核心问题：

        当前已经完成了什么？
        当前拿到了什么事实？
        还缺什么？
        下一步应该由谁完成？
        应该把什么具体任务交给这个 Specialist？
    """

    # ----------------------------------------------------------
    # Memories
    # ----------------------------------------------------------

    memories_text = (
        "\n".join(
            f"- {memory}"
            for memory in retrieved_memories
        )
        if retrieved_memories
        else "No relevant long-term memories."
    )

    # ----------------------------------------------------------
    # Completed Agents
    # ----------------------------------------------------------

    completed_agents_text = (
        "\n".join(
            f"- {agent}"
            for agent in completed_agents
        )
        if completed_agents
        else "None"
    )

    # ----------------------------------------------------------
    # Tool Results
    # ----------------------------------------------------------

    tool_results_text = (
        _format_tool_results_for_prompt(
            tool_results
        )
    )

    # ----------------------------------------------------------
    # Current Handoff Task
    # ----------------------------------------------------------

    current_handoff_task_text = (
        handoff_task
        if handoff_task
        else "None"
    )

    return f"""
You are the Supervisor of an enterprise multi-agent system.

Your role is WORKFLOW PLANNING and TASK DELEGATION.

You MUST decide what should happen NEXT based on:
    1. the user's original request,
    2. completed specialist agents,
    3. tool execution results,
    4. current workflow state,
    5. current handoff task.

You MUST NOT execute tools.
You MUST NOT generate the final user answer.
You MUST ONLY return a structured routing decision.

============================================================
AVAILABLE SPECIALIST AGENTS
============================================================

1. knowledge_agent

Responsibilities:
- Enterprise internal knowledge retrieval
- Internal documents
- Enterprise policies
- Product documentation
- Internal RAG
- Internal knowledge base questions

Tool:
- rag_search


2. operations_agent

Responsibilities:
- Service health
- Service status
- Infrastructure state
- Operational diagnostics
- Read-only operational queries

Tool:
- get_service_health


3. ticket_agent

Responsibilities:
- Query tickets
- Create tickets
- Update tickets
- Incident management
- Ticket lifecycle operations

Tools:
- get_ticket
- create_ticket
- update_ticket


4. research_agent

Responsibilities:
- External web search
- Public technical documentation
- Current public information
- External research
- News
- Public internet information

Tool:
- web_search


============================================================
CORE MULTI-AGENT PLANNING RULE
============================================================

A user request may require MULTIPLE specialist agents.

Do NOT stop after the first specialist if the user's
overall request still requires another specialist.

Example:

User:
"Check payment-service health. If it is degraded,
create a P1 incident."

Correct workflow:

1. operations_agent
   → get_service_health

2. inspect the tool result

3. If service is degraded:
   → ticket_agent

4. ticket_agent
   → create_ticket

5. after the ticket operation succeeds:
   → end

Therefore, the Supervisor must reason over previous
Tool Results and completed specialists.

============================================================
TASK HANDOFF RULE
============================================================

When selecting a specialist, you MUST create a concise
task description specifically for that Specialist.

The "task" field is NOT the entire user query.

It should describe only the next concrete responsibility.

Example:

User:
"Check payment-service health. If degraded, create a P1
incident."

First handoff:

next_agent:
    "operations_agent"

task:
    "Check the current health status of payment-service
     and determine whether it is abnormal."

Second handoff:

next_agent:
    "ticket_agent"

task:
    "Create a P1 Incident for the degraded payment-service
     using the health-check result."

When the workflow ends:

next_agent:
    "end"

workflow_complete:
    true

task:
    can be empty or omitted semantically, because no
    Specialist is being handed a new task.

============================================================
IMPORTANT
============================================================

Do NOT select an agent merely because its tool exists.

Select the agent based on the user's ACTUAL remaining task.

Do NOT repeat a specialist unnecessarily.

If a required specialist has already completed its part
of the task and the result is sufficient, move to the next
required specialist.

If the overall user request has been completely satisfied,
return:

    next_agent = "end"
    workflow_complete = true

Otherwise:

    next_agent = "<specialist>"
    workflow_complete = false

    task = "<specific handoff task>"


============================================================
SPECIALIST SELECTION RULES
============================================================

Use knowledge_agent for:
- internal enterprise knowledge
- internal documents
- policies
- internal product information

Use operations_agent for:
- current service health
- operational status
- infrastructure diagnostics

Use ticket_agent for:
- creating incidents
- updating incidents
- querying tickets
- ticket lifecycle actions

Use research_agent for:
- external web information
- public documentation
- current external information

Use end ONLY when:
- the entire user request is satisfied, OR
- no meaningful specialist action remains.


============================================================
CURRENT USER REQUEST
============================================================

{user_query}


============================================================
RELEVANT LONG-TERM MEMORIES
============================================================

{memories_text}


============================================================
COMPLETED SPECIALIST AGENTS
============================================================

{completed_agents_text}


============================================================
TOOL EXECUTION RESULTS
============================================================

{tool_results_text}


============================================================
CURRENT AGENT
============================================================

{current_agent or "none"}


============================================================
CURRENT HANDOFF TASK
============================================================

{current_handoff_task_text}


============================================================
CURRENT TASK STATUS
============================================================

{task_status}


============================================================
PLANNING INSTRUCTIONS
============================================================

Before selecting the next agent, determine:

1. What has already been completed?
2. What information has already been obtained?
3. What actions are still required?
4. Which Specialist is responsible for the remaining action?
5. What exact task should be delegated to that Specialist?
6. Is the overall user request fully satisfied?

Remember:

A Specialist completing its own subtask does NOT necessarily
mean the entire Workflow is complete.

The "task" field must describe the next concrete Specialist
responsibility.

Return ONLY valid JSON.

Required format:

{{
  "next_agent": "knowledge_agent | operations_agent | ticket_agent | research_agent | end",
  "reason": "Explain what remains and why the selected agent is the correct next step.",
  "task": "Concise concrete task description for the selected specialist agent.",
  "workflow_complete": true
}}

For any specialist:

    workflow_complete MUST be false.

For "end":

    workflow_complete MUST be true.

When next_agent is "end", task may be an empty string.
""".strip()


# ==============================================================
# Tool Result Formatting
# ==============================================================

def _format_tool_results_for_prompt(
    tool_results: list[Any],
) -> str:
    """
    将 Tool Execution Facts 转换成 Supervisor 可理解的文本。

    Supervisor 只需要知道：

        tool_name
        success / failure
        result content

    对 content 做长度限制，避免 Prompt 无限膨胀。
    """

    if not tool_results:
        return "No tool has been executed yet."

    lines: list[str] = []

    for index, result in enumerate(
        tool_results,
        start=1,
    ):
        if not isinstance(
            result,
            dict,
        ):
            lines.append(
                f"- Tool Result #{index}: {result}"
            )
            continue

        tool_name = result.get(
            "tool_name",
            "unknown_tool",
        )

        error = result.get(
            "error",
            False,
        )

        error_type = result.get(
            "error_type",
        )

        error_message = result.get(
            "error_message",
        )

        content = result.get(
            "content",
        )

        status = (
            "FAILED"
            if error
            else "SUCCESS"
        )

        lines.append(
            f"- Tool: {tool_name}"
        )

        lines.append(
            f"  Status: {status}"
        )

        if error:
            if error_type:
                lines.append(
                    f"  Error Type: {error_type}"
                )

            if error_message:
                lines.append(
                    f"  Error: {error_message}"
                )

        if content is not None:
            content_text = str(
                content
            ).strip()

            max_length = 6000

            if len(content_text) > max_length:
                content_text = (
                    content_text[:max_length]
                    + "...[truncated]"
                )

            lines.append(
                f"  Content: {content_text}"
            )

    return "\n".join(
        lines
    )


# ==============================================================
# Latest User Query Extraction
# ==============================================================

def _extract_latest_user_query(
    messages: list[Any],
) -> str:
    """
    从 Graph Message State 中取得最后一条用户消息。

    支持：

    1. LangChain BaseMessage
    2. OpenAI-style dict
    """

    for message in reversed(
        messages
    ):
        # ======================================================
        # 1. LangChain BaseMessage
        # ======================================================

        if isinstance(
            message,
            BaseMessage,
        ):
            message_type = getattr(
                message,
                "type",
                None,
            )

            if message_type != "human":
                continue

            content = message.content

            text = _extract_text_content(
                content
            )

            if text:
                return text

            continue

        # ======================================================
        # 2. OpenAI-style message dict
        # ======================================================

        if isinstance(
            message,
            dict,
        ):
            role = message.get(
                "role"
            )

            if role != "user":
                continue

            content = message.get(
                "content"
            )

            text = _extract_text_content(
                content
            )

            if text:
                return text

    return ""


# ==============================================================
# Content Extraction
# ==============================================================

def _extract_text_content(
    content: Any,
) -> str:
    """
    从不同类型的 message.content 中提取纯文本。

    支持：

    1. str
    2. LangChain/OpenAI content blocks
    3. 字符串列表
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


# ==============================================================
# Model Response Content Extraction
# ==============================================================

def _extract_response_content(
    content: Any,
) -> str:
    """
    兼容不同 ChatModel response.content 格式。
    """

    return _extract_text_content(
        content
    )


# ==============================================================
# Supervisor Decision Parsing
# ==============================================================

def _parse_decision(
    raw_content: str,
) -> SupervisorDecision:
    """
    JSON Parse + Pydantic Validation。

    对 task 做防御性归一化，避免模型返回 task=""
    时导致整个 Supervisor Workflow 失败。
    """

    if not raw_content:
        raise ValueError(
            "Supervisor returned empty content."
        )

    cleaned = raw_content.strip()

    # ----------------------------------------------------------
    # Defensive Markdown JSON Fence Handling
    # ----------------------------------------------------------

    if cleaned.startswith(
        "```"
    ):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

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
        payload = json.loads(
            cleaned
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Supervisor returned invalid JSON."
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Supervisor decision must be a JSON object."
        )

    # ----------------------------------------------------------
    # task 防御性归一化
    # ----------------------------------------------------------

    task = payload.get(
        "task"
    )

    if (
        task is None
        or (
            isinstance(
                task,
                str,
            )
            and not task.strip()
        )
    ):
        payload = {
            **payload,
            "task": (
                "Continue the workflow based on "
                "the selected agent."
            ),
        }

    # ----------------------------------------------------------
    # Pydantic Validation
    # ----------------------------------------------------------

    try:
        decision = (
            SupervisorDecision.model_validate(
                payload
            )
        )

    except ValidationError as exc:
        raise ValueError(
            f"Invalid Supervisor decision: {exc}"
        ) from exc

    # ----------------------------------------------------------
    # Semantic Validation
    # ----------------------------------------------------------

    if (
        decision.next_agent == "end"
        and not decision.workflow_complete
    ):
        raise ValueError(
            "Supervisor returned next_agent='end' "
            "but workflow_complete=false."
        )

    if (
        decision.next_agent != "end"
        and decision.workflow_complete
    ):
        raise ValueError(
            "Supervisor selected a specialist while "
            "workflow_complete=true."
        )

    return decision

