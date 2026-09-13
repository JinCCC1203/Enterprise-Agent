from __future__ import annotations

import json
import re
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
        下一步执行的 Specialist，或 end。

    reason:
        当前路由决策原因。

    task:
        给下一 Specialist 的具体任务。

    workflow_complete:
        LLM 对整个 Workflow 是否完成的判断。

    注意：
        workflow_complete 只是 LLM 的候选判断，
        最终是否允许 end，由 Runtime Completion Guard 再次校验。
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
            "True only when all required business actions "
            "have actually been completed successfully."
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
        归一化 LLM 返回的空 task。
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

    Supervisor 负责：

        Workflow Planning
        Specialist Routing
        Task Delegation
        Workflow Completion Decision

    Supervisor 不：

        - 执行 Tool
        - 生成最终用户答案
        - 修改业务数据
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

        recovery_status = state.get(
            "recovery_status",
        )

        recovery_attempts = state.get(
            "recovery_attempts",
            0,
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
                "workflow_complete": False,
                "task_status": "failed",
                "handoff_reason": (
                    "No user query was found."
                ),
                "handoff_task": None,
                "current_agent": None,
            }

        # ----------------------------------------------------------
        # 2. Recovery 已明确耗尽
        #
        # terminal failure 不允许 Supervisor 再次规划。
        # ----------------------------------------------------------

        if recovery_status == "failed":
            return {
                "next_agent": "end",
                "current_agent": None,
                "handoff_task": None,
                "workflow_complete": False,
                "task_status": "failed",
                "handoff_reason": (
                    "Workflow recovery has been exhausted. "
                    "The workflow must terminate in a failed state."
                ),
            }

        # ----------------------------------------------------------
        # 3. Recovery reroute 模式
        # ----------------------------------------------------------

        is_recovery_reroute = (
            recovery_status == "reroute"
        )

        # ----------------------------------------------------------
        # 4. 普通执行阶段 Failure Guard
        # ----------------------------------------------------------

        if (
            not is_recovery_reroute
            and _has_unresolved_failure(
                state
            )
        ):
            failed_tool = state.get(
                "last_failed_tool"
            )

            failed_node = state.get(
                "last_failed_node"
            )

            error = state.get(
                "error"
            )

            return {
                "next_agent": "end",
                "current_agent": None,
                "handoff_task": None,
                "workflow_complete": False,
                "task_status": "failed",
                "handoff_reason": (
                    "The workflow cannot be completed because "
                    "a required operation failed and no successful "
                    "tool result confirms completion."
                    f" Failed node={failed_node!r}, "
                    f"failed tool={failed_tool!r}, "
                    f"error={error!r}."
                ),
            }

        # ----------------------------------------------------------
        # 5. Recovery reroute 上下文
        # ----------------------------------------------------------

        recovery_context = ""

        if is_recovery_reroute:

            failed_node = state.get(
                "last_failed_node"
            )

            failed_tool = state.get(
                "last_failed_tool"
            )

            error = state.get(
                "error"
            )

            recovery_reason = state.get(
                "recovery_reason"
            )

            recovery_context = f"""
IMPORTANT RECOVERY CONTEXT

This Supervisor invocation was triggered by
Workflow Recovery.

Recovery status:
    reroute

Recovery attempt count:
    {recovery_attempts}

Previous failed node:
    {failed_node or "unknown"}

Previous failed tool:
    {failed_tool or "unknown"}

Previous error:
    {error or "unknown"}

Recovery reason:
    {recovery_reason or "unknown"}

IMPORTANT:

The previous failure has already been accepted by the
Recovery subsystem and the workflow is now entering
a NEW planning cycle.

Do NOT terminate the workflow merely because the previous
error still exists in State.

You MUST re-plan the workflow.

If the original failed Specialist is still required,
route back to that Specialist and let it retry the
required business action.

A recovery reroute is NOT a successful completion.

workflow_complete MUST remain false until the required
business operation succeeds.
""".strip()

        # ----------------------------------------------------------
        # 6. 构造 Supervisor Prompt
        # ----------------------------------------------------------

        prompt = _build_supervisor_prompt(
            user_query=user_query,
            retrieved_memories=retrieved_memories,
            completed_agents=completed_agents,
            tool_results=tool_results,
            current_agent=current_agent,
            handoff_task=handoff_task,
            task_status=task_status,
            recovery_context=recovery_context,
        )

        # ----------------------------------------------------------
        # 7. 调用 LLM
        # ----------------------------------------------------------

        response = await json_model.ainvoke(
            prompt
        )

        # ----------------------------------------------------------
        # 8. 提取模型输出
        # ----------------------------------------------------------

        raw_content = _extract_response_content(
            response.content
        )

        # ----------------------------------------------------------
        # 9. Parse + Validation
        # ----------------------------------------------------------

        decision = _parse_decision(
            raw_content
        )

        # ----------------------------------------------------------
        # 10. 调试信息
        #
        # 可以保留，确认 Supervisor 第二次规划时
        # 实际拿到了哪些 State。
        # ----------------------------------------------------------

        print(
            "[Supervisor Decision]",
            {
                "next_agent": decision.next_agent,
                "workflow_complete": decision.workflow_complete,
                "current_agent": current_agent,
                "completed_agents": completed_agents,
                "task_status": task_status,
                "recovery_status": recovery_status,
                "handoff_task": handoff_task,
            },
        )

        # ----------------------------------------------------------
        # 11. Recovery Reroute 程序级保护
        # ----------------------------------------------------------

        if (
            is_recovery_reroute
            and decision.next_agent == "end"
        ):

            recovery_agent = (
                _get_recovery_replan_agent(
                    state
                )
            )

            if recovery_agent is not None:

                recovery_task = (
                    _build_recovery_replan_task(
                        state
                    )
                )

                return {
                    "next_agent": recovery_agent,
                    "current_agent": recovery_agent,
                    "handoff_task": recovery_task,
                    "workflow_complete": False,
                    "task_status": "running",
                    "handoff_reason": (
                        "Recovery requested re-planning, "
                        "but Supervisor proposed end before "
                        "the failed business action was retried. "
                        f"Returning control to {recovery_agent!r}."
                    ),
                }

            return {
                "next_agent": "end",
                "current_agent": None,
                "handoff_task": None,
                "workflow_complete": False,
                "task_status": "failed",
                "handoff_reason": (
                    "Recovery requested re-planning, but no "
                    "recoverable Specialist could be determined."
                ),
            }

        # ----------------------------------------------------------
        # 12. Completion Guard
        #
        # 重要：
        #
        # 不再提前使用 state.workflow_complete=True
        # 直接结束。
        #
        # 只要 Supervisor 想返回 end，
        # 就必须检查用户任务是否还有 pending business action。
        # ----------------------------------------------------------

        if decision.next_agent == "end":

            pending_route = (
                _get_pending_business_action_route(
                    user_query=user_query,
                    tool_results=tool_results,
                    completed_agents=completed_agents,
                    current_agent=current_agent,
                )
            )

            if pending_route is not None:

                print(
                    "[Supervisor Completion Guard]",
                    {
                        "action": "reroute",
                        "target": pending_route.get(
                            "next_agent"
                        ),
                        "reason": pending_route.get(
                            "handoff_reason"
                        ),
                    },
                )

                return pending_route

            # ------------------------------------------------------
            # 12.1 unresolved failure Guard
            # ------------------------------------------------------

            if _has_unresolved_failure(
                state
            ):

                failed_tool = state.get(
                    "last_failed_tool"
                )

                return {
                    "next_agent": "end",
                    "current_agent": None,
                    "handoff_task": None,
                    "workflow_complete": False,
                    "task_status": "failed",
                    "handoff_reason": (
                        "Supervisor proposed end, but the workflow "
                        "still contains an unresolved failure"
                        f" for tool {failed_tool!r}."
                    ),
                }

            # ------------------------------------------------------
            # 12.2 真正允许结束
            # ------------------------------------------------------

            return {
                "next_agent": "end",
                "current_agent": None,
                "handoff_task": None,
                "workflow_complete": True,
                "task_status": "completed",
                "handoff_reason": (
                    "The workflow is complete and no required "
                    "business action remains pending."
                ),
            }

        # ----------------------------------------------------------
        # 13. Specialist 路由
        #
        # Specialist 决不能声明整个 Workflow 已完成。
        # ----------------------------------------------------------

        return {
            "next_agent": decision.next_agent,
            "current_agent": decision.next_agent,
            "handoff_task": decision.task,
            "workflow_complete": False,
            "task_status": "running",
            "handoff_reason": decision.reason,
        }

    return supervisor_node


# ==============================================================
# Pending Business Action Guard
# ==============================================================

def _get_pending_business_action_route(
    *,
    user_query: str,
    tool_results: list[Any],
    completed_agents: list[str],
    current_agent: str | None,
) -> dict[str, Any] | None:
    """
    程序级 Workflow Completion Guard。

    当前重点覆盖：

        health check
        +
        degraded
        +
        create incident / ticket

    核心原则：

        LLM 可以规划，
        但不能绕过已知的业务事实直接结束 Workflow。
    """

    # ----------------------------------------------------------
    # 1. 用户是否明确要求创建 Ticket / Incident
    # ----------------------------------------------------------

    requires_ticket_creation = (
        _query_requires_ticket_creation(
            user_query
        )
    )

    if not requires_ticket_creation:
        return None

    # ----------------------------------------------------------
    # 2. create_ticket 已经成功
    # ----------------------------------------------------------

    if _has_successful_tool_result(
        tool_results,
        "create_ticket",
    ):
        return None

    # ----------------------------------------------------------
    # 3. 用户已经明确 reject
    #
    # 不应该自动再次发起副作用操作。
    # ----------------------------------------------------------

    if _was_tool_rejected(
        tool_results,
        "create_ticket",
    ):
        return None

    # ----------------------------------------------------------
    # 4. 如果是条件性 Ticket 创建任务
    # ----------------------------------------------------------

    conditional_health_ticket = (
        _query_requires_conditional_ticket(
            user_query
        )
    )

    if conditional_health_ticket:

        health_result = (
            _get_latest_service_health_result(
                tool_results
            )
        )

        # ------------------------------------------------------
        # 4.1 健康检查尚未完成
        # ------------------------------------------------------

        if health_result is None:

            return {
                "next_agent": "operations_agent",
                "current_agent": "operations_agent",
                "handoff_task": (
                    "Check the current health status of "
                    "payment-service and determine whether "
                    "the service is degraded or abnormal."
                ),
                "workflow_complete": False,
                "task_status": "running",
                "handoff_reason": (
                    "The user's conditional ticket workflow "
                    "requires a service health result before "
                    "the downstream ticket action can be evaluated."
                ),
            }

        # ------------------------------------------------------
        # 4.2 条件没有成立
        # ------------------------------------------------------

        if not _health_result_is_degraded(
            health_result
        ):
            return None

    # ----------------------------------------------------------
    # 5. 条件已经成立：
    #
    #     create_ticket 尚未成功
    #
    #     → ticket_agent
    # ----------------------------------------------------------

    return {
        "next_agent": "ticket_agent",
        "current_agent": "ticket_agent",
        "handoff_task": (
            "Create the required P1 Incident for the degraded "
            "payment-service described in the user's original "
            "request. Use create_ticket and only report completion "
            "after the tool returns a successful result."
        ),
        "workflow_complete": False,
        "task_status": "running",
        "handoff_reason": (
            "The user requested a ticket creation action that "
            "has not yet succeeded. The service health result "
            "confirms the degraded condition, so ticket_agent "
            "must execute the remaining business action."
        ),
    }


# ==============================================================
# Query Intent Detection
# ==============================================================

def _query_requires_ticket_creation(
    user_query: str,
) -> bool:
    """
    判断用户请求是否明确要求创建 Incident / Ticket。

    这里仅作为 Completion Guard 的防御逻辑，
    不替代 Supervisor 的自然语言理解。
    """

    normalized = (
        user_query
        .strip()
        .lower()
    )

    if not normalized:
        return False

    patterns = [
        # ------------------------------------------------------
        # Chinese
        # ------------------------------------------------------

        r"创建.{0,30}(工单|incident)",
        r"新建.{0,30}(工单|incident)",
        r"生成.{0,30}(工单|incident)",
        r"建立.{0,30}(工单|incident)",
        r"开.{0,10}(工单|incident)",
        r"开一个.{0,30}(工单|incident)",

        # ------------------------------------------------------
        # English
        # ------------------------------------------------------

        r"\bcreate\b.{0,40}\b(ticket|incident)\b",
        r"\bcreate\b.{0,40}\bincident\b",
        r"\bopen\b.{0,30}\b(ticket|incident)\b",
        r"\braise\b.{0,30}\b(ticket|incident)\b",
    ]

    return any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in patterns
    )


def _query_requires_conditional_ticket(
    user_query: str,
) -> bool:
    """
    判断是否存在：

        条件成立
            ↓
        创建 Ticket / Incident

    例如：

        如果服务为 degraded，请创建 P1 Incident。

        If the service is degraded, create a ticket.
    """

    normalized = (
        user_query
        .strip()
        .lower()
    )

    if not normalized:
        return False

    if not _query_requires_ticket_creation(
        user_query
    ):
        return False

    condition_patterns = [
        # Chinese
        r"如果",
        r"若",
        r"当",
        r"一旦",
        r"状态.*为",
        r"异常",
        r"故障",

        # English
        r"\bif\b",
        r"\bwhen\b",
        r"\bonce\b",
        r"\bdegraded\b",
        r"\bunhealthy\b",
        r"\babnormal\b",
    ]

    return any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in condition_patterns
    )


# ==============================================================
# Tool Result Helpers
# ==============================================================

def _has_successful_tool_result(
    tool_results: list[Any],
    tool_name: str,
) -> bool:
    """
    判断指定 Tool 是否已经成功执行。
    """

    for result in tool_results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        if result.get(
            "tool_name"
        ) != tool_name:
            continue

        if result.get(
            "error",
            False,
        ):
            continue

        content = result.get(
            "content"
        )

        if content is None:
            return True

        content_text = str(
            content
        ).strip()

        # ------------------------------------------------------
        # MCP structured result 明确表示业务失败
        # ------------------------------------------------------

        if re.search(
            r'"success"\s*:\s*false',
            content_text,
            flags=re.IGNORECASE,
        ):
            continue

        # ------------------------------------------------------
        # Tool execution 成功
        # ------------------------------------------------------

        return True

    return False


def _was_tool_rejected(
    tool_results: list[Any],
    tool_name: str,
) -> bool:
    """
    判断 Tool 是否被 Human-in-the-Loop 明确拒绝。
    """

    for result in tool_results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        if result.get(
            "tool_name"
        ) != tool_name:
            continue

        content = result.get(
            "content"
        )

        if content is None:
            continue

        content_text = str(
            content
        ).lower()

        if (
            "user rejected the tool call"
            in content_text
            or "tool was not executed"
            in content_text
        ):
            return True

    return False


def _get_latest_service_health_result(
    tool_results: list[Any],
) -> dict[str, Any] | None:
    """
    获取最近一次 get_service_health 的成功结果。

    当前 tool_results 中保存的是字符串化 MCP Tool content，
    因此这里提取关键 status 字段。
    """

    for result in reversed(
        tool_results
    ):

        if not isinstance(
            result,
            dict,
        ):
            continue

        if result.get(
            "tool_name"
        ) != "get_service_health":
            continue

        if result.get(
            "error",
            False,
        ):
            continue

        content = result.get(
            "content"
        )

        if content is None:
            continue

        content_text = str(
            content
        )

        status_match = re.search(
            r'"status"\s*:\s*"([^"]+)"',
            content_text,
            flags=re.IGNORECASE,
        )

        if not status_match:
            continue

        status = (
            status_match
            .group(1)
            .strip()
            .lower()
        )

        return {
            "status": status,
            "content": content_text,
            "raw_result": result,
        }

    return None


def _health_result_is_degraded(
    health_result: dict[str, Any],
) -> bool:
    """
    判断服务健康结果是否为 degraded。
    """

    status = health_result.get(
        "status"
    )

    if not isinstance(
        status,
        str,
    ):
        return False

    return (
        status.strip().lower()
        == "degraded"
    )


# ==============================================================
# Unresolved Failure Detection
# ==============================================================

def _has_unresolved_failure(
    state: EnterpriseAgentState,
) -> bool:
    """
    判断当前 Workflow 是否仍存在未解决失败。

    关键原则：

        failed_tool = create_ticket
        +
        没有 create_ticket 成功结果
        =
        unresolved failure
    """

    error = (
        state.get(
            "error"
        ) or ""
    ).strip()

    failed_tool = state.get(
        "last_failed_tool"
    )

    recovery_status = state.get(
        "recovery_status"
    )

    # ----------------------------------------------------------
    # Recovery 已经 terminal failure
    # ----------------------------------------------------------

    if recovery_status == "failed":
        return True

    # ----------------------------------------------------------
    # 没有失败信息
    # ----------------------------------------------------------

    if (
        not error
        and not failed_tool
    ):
        return False

    # ----------------------------------------------------------
    # 有 error 但没有具体 Tool
    # ----------------------------------------------------------

    if not failed_tool:
        return True

    # ----------------------------------------------------------
    # 检查 Tool Results
    # ----------------------------------------------------------

    tool_results = state.get(
        "tool_results",
        [],
    )

    for result in tool_results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        if result.get(
            "tool_name"
        ) != failed_tool:
            continue

        if not result.get(
            "error",
            False,
        ):
            return False

    return True


# ==============================================================
# Recovery Re-plan Agent
# ==============================================================

def _get_recovery_replan_agent(
    state: EnterpriseAgentState,
) -> str | None:
    """
    Recovery reroute 后确定默认重试 Specialist。
    """

    failed_node = state.get(
        "last_failed_node"
    )

    if failed_node in {
        "knowledge_agent",
        "operations_agent",
        "ticket_agent",
        "research_agent",
    }:
        return failed_node

    current_agent = state.get(
        "current_agent"
    )

    if current_agent in {
        "knowledge_agent",
        "operations_agent",
        "ticket_agent",
        "research_agent",
    }:
        return current_agent

    return None


# ==============================================================
# Recovery Re-plan Task
# ==============================================================

def _build_recovery_replan_task(
    state: EnterpriseAgentState,
) -> str:
    """
    构造 Recovery reroute 后重新交给 Specialist 的任务。
    """

    failed_tool = state.get(
        "last_failed_tool"
    )

    previous_handoff_task = state.get(
        "handoff_task"
    )

    if previous_handoff_task:
        base_task = (
            previous_handoff_task
        )
    else:
        base_task = (
            "Retry the required business action "
            "from the original workflow."
        )

    if failed_tool:
        return (
            f"{base_task} "
            f"The previous attempt using tool "
            f"'{failed_tool}' failed. Retry the required "
            f"business action and only report completion "
            f"after the tool succeeds."
        )

    return (
        f"{base_task} "
        "The previous attempt failed. Retry the required "
        "business action and only report completion after "
        "the operation succeeds."
    )


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
    recovery_context: str = "",
) -> str:
    """
    构造 Multi-Specialist Supervisor Prompt。
    """

    memories_text = (
        "\n".join(
            f"- {memory}"
            for memory in retrieved_memories
        )
        if retrieved_memories
        else "No relevant long-term memories."
    )

    completed_agents_text = (
        "\n".join(
            f"- {agent}"
            for agent in completed_agents
        )
        if completed_agents
        else "None"
    )

    tool_results_text = (
        _format_tool_results_for_prompt(
            tool_results
        )
    )

    current_handoff_task_text = (
        handoff_task
        if handoff_task
        else "None"
    )

    recovery_context_text = (
        recovery_context
        if recovery_context
        else "No active Workflow Recovery reroute."
    )

    return f"""
You are the Supervisor of an enterprise multi-agent system.

Your role is WORKFLOW PLANNING and TASK DELEGATION.

You MUST decide what should happen NEXT based on:

    1. the user's original request,
    2. completed specialist agents,
    3. tool execution results,
    4. current workflow state,
    5. current handoff task,
    6. whether any required operation has actually succeeded,
    7. whether Workflow Recovery has requested re-planning.

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
CRITICAL WORKFLOW COMPLETION RULE
============================================================

A Specialist completing its own subtask does NOT mean the
entire workflow is complete.

The workflow is complete ONLY when every required business
action from the user's request has actually succeeded.

Example:

"Check payment-service health. If degraded, create P1 incident."

Correct:

operations_agent
→ get_service_health
→ degraded

ticket_agent
→ create_ticket
→ successful result

then:

end
→ workflow_complete=true


Incorrect:

operations_agent
→ get_service_health
→ degraded

then:

end

The workflow is NOT complete because the user still requires
the conditional ticket creation action.

Likewise:

ticket_agent
→ create_ticket
→ failed

then:

end

is also NOT a successful completion.


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

If create_ticket fails:
    DO NOT end successfully.


============================================================
RECOVERY REROUTE RULE
============================================================

If Workflow Recovery reports:

    recovery_status = "reroute"

then:

1. The previous failure has already been classified by Recovery.
2. The Supervisor MUST start a new planning cycle.
3. The Supervisor MUST NOT terminate merely because old
   error information remains in State.
4. The Supervisor should normally return control to the
   Specialist associated with the failed business operation.
5. workflow_complete MUST remain false.
6. The Specialist must retry the failed business action.
7. The workflow can end successfully only after the required
   operation succeeds.

A Recovery reroute is NOT a successful completion.


============================================================
TASK HANDOFF RULE
============================================================

When selecting a specialist, create a concise task for it.

The "task" field is NOT the entire user query.

It describes only the next concrete responsibility.

Example:

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
- every required business action has succeeded, OR
- no meaningful specialist action remains.

Never select end merely because a specialist has been
delegated or because an operation was attempted.


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
WORKFLOW RECOVERY CONTEXT
============================================================

{recovery_context_text}


============================================================
PLANNING INSTRUCTIONS
============================================================

Before selecting the next agent:

1. What has already been completed?
2. What information has already been obtained?
3. Which required actions have actually succeeded?
4. Which required actions are still pending?
5. Which required actions failed?
6. Is a Recovery reroute currently active?
7. If Recovery reroute is active, which Specialist should
   retry the failed business action?
8. What action is still required?
9. Which Specialist is responsible for that action?
10. Has the entire user request actually been satisfied?

IMPORTANT:

A Specialist completing its own subtask does NOT necessarily
mean the entire Workflow is complete.

A Tool being attempted does NOT mean its business action
succeeded.

Only successful Tool Execution Facts can establish that a
required Tool action has succeeded.

If the user's request contains a conditional business action,
such as:

    "If service is degraded, create a P1 Incident."

then the condition MUST be evaluated from the tool result,
and the downstream business action MUST still be executed
when the condition is satisfied.

Do NOT return end while a required downstream business action
is still pending.

If recovery_status = "reroute", do NOT return end unless
the required business operation has already succeeded after
the Recovery action.

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
    将 Tool Execution Facts 转成 Supervisor Prompt。
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
            "error_type"
        )

        error_message = result.get(
            "error_message"
        )

        content = result.get(
            "content"
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
    提取最后一条 Human/User Message。
    """

    for message in reversed(
        messages
    ):
        if isinstance(
            message,
            BaseMessage,
        ):
            if getattr(
                message,
                "type",
                None,
            ) != "human":
                continue

            text = _extract_text_content(
                message.content
            )

            if text:
                return text

            continue

        if isinstance(
            message,
            dict,
        ):
            if message.get(
                "role"
            ) != "user":
                continue

            text = _extract_text_content(
                message.get(
                    "content"
                )
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
    从 message.content 提取纯文本。
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
    提取 ChatModel Response Content。
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
    """

    if not raw_content:
        raise ValueError(
            "Supervisor returned empty content."
        )

    cleaned = raw_content.strip()

    # ----------------------------------------------------------
    # Markdown JSON Fence
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