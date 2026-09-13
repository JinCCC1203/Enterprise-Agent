from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphInterrupt
from langgraph.runtime import Runtime

from middlewares.dynamic_tools import (
    DynamicToolMiddleware,
)
from policies.permission import PermissionPolicy
from tools_manager.registry import ToolRegistry
from tools_manager.tool_exposure import (
    PermissionBasedToolExposure,
)

from workflow.state import (
    EnterpriseAgentContext,
    EnterpriseAgentState,
)
from workflow.utils.execution_errors import (
    extract_tool_execution_error,
)
from workflow.utils.execution_facts import (
    collect_tool_results,
)


# ==============================================================
# Ticket Tool Scope
# ==============================================================

TICKET_TOOL_SCOPE = frozenset(
    {
        "get_ticket",
        "create_ticket",
        "update_ticket",
    }
)


# ==============================================================
# Ticket Agent System Prompt
# ==============================================================

TICKET_AGENT_SYSTEM_PROMPT = """
You are the Ticket Specialist Agent in an enterprise
multi-agent workflow.

Your responsibility is to execute the ticket-management
task delegated by the Supervisor.

Available ticket tools:

- get_ticket
- create_ticket
- update_ticket

IMPORTANT EXECUTION RULES:

1. You MUST follow the Supervisor's current handoff task.

2. If the handoff task requires querying a ticket,
   call get_ticket.

3. If the handoff task requires creating an incident or
   ticket, you MUST call create_ticket.

4. If the handoff task requires updating an existing ticket,
   you MUST call update_ticket.

5. Do NOT claim that a ticket was created or updated unless
   the corresponding tool call actually succeeded.

6. A natural-language response alone does NOT mean the
   delegated task has been completed.

7. For operations with external side effects, execute the
   appropriate tool and allow the surrounding Agent Runtime
   to handle Permission Policy, Risk Policy, Retry, and
   Human-in-the-Loop approval.

8. When the required tool operation succeeds, summarize
   the actual tool result and stop.

9. Do not perform unrelated ticket operations.

10. Treat the Supervisor handoff task as the current task
    boundary for this Specialist.
""".strip()


# ==============================================================
# Create Ticket Agent
# ==============================================================

def create_ticket_agent(
    *,
    model: ChatOpenAI,
    registry: ToolRegistry,
    permission_policy: PermissionPolicy,
    middleware: list[Any] | None = None,
):
    """
    创建 Ticket Specialist Agent。

    Scope:
        get_ticket
        create_ticket
        update_ticket

    Runtime responsibilities:
        - Supervisor handoff
        - Dynamic Tool Exposure
        - Permission Policy
        - Retry
        - Tool Error Handling
        - HITL
        - Tool Result verification
    """

    # ==============================================================
    # 1. Specialist Registry View
    # ==============================================================

    registry_view = registry.create_view(
        TICKET_TOOL_SCOPE
    )

    # ==============================================================
    # 2. Tool Exposure
    # ==============================================================

    tool_exposure = (
        PermissionBasedToolExposure(
            registry_view=registry_view,
            permission_policy=permission_policy,
        )
    )

    # ==============================================================
    # 3. Dynamic Tool Middleware
    # ==============================================================

    dynamic_tools = DynamicToolMiddleware(
        tool_exposure=tool_exposure,
        agent_name="ticket_agent",
    )

    # ==============================================================
    # 4. Middleware
    # ==============================================================

    agent_middleware = [
        dynamic_tools,
        *(middleware or []),
    ]

    # ==============================================================
    # 5. Scoped Tools
    # ==============================================================

    tools: list[BaseTool] = (
        registry_view.get_all_tools()
    )

    # ==============================================================
    # 6. LangChain Agent
    # ==============================================================

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=agent_middleware,
        context_schema=EnterpriseAgentContext,
    )

    # ==============================================================
    # 7. LangGraph Node
    # ==============================================================

    async def ticket_agent_node(
        state: EnterpriseAgentState,
        runtime: Runtime[EnterpriseAgentContext],
        config: RunnableConfig,
    ) -> dict[str, Any]:

        messages = list(
            state.get(
                "messages",
                [],
            )
        )

        retrieved_memories = state.get(
            "retrieved_memories",
            [],
        )

        handoff_task = state.get(
            "handoff_task",
        )

        # ==========================================================
        # 8. Build Context
        # ==========================================================

        memory_context = _build_memory_context(
            retrieved_memories
        )

        handoff_context = _build_handoff_context(
            handoff_task
        )

        # ==========================================================
        # 9. Build Specialist Messages
        #
        # 顺序：
        #
        # System
        # ↓
        # Historical conversation
        # ↓
        # Memory context
        # ↓
        # Supervisor handoff task
        #
        # handoff_task 放最后，确保 Specialist 最近看到的
        # 明确目标就是当前 Supervisor 委派任务。
        # ==========================================================

        agent_messages: list[Any] = []

        agent_messages.append(
            SystemMessage(
                content=TICKET_AGENT_SYSTEM_PROMPT
            )
        )

        agent_messages.extend(
            messages
        )

        if memory_context:
            agent_messages.append(
                HumanMessage(
                    content=memory_context
                )
            )

        if handoff_context:
            agent_messages.append(
                HumanMessage(
                    content=handoff_context
                )
            )

        # ==========================================================
        # 10. Execute Agent
        # ==========================================================

        try:
            result = await agent.ainvoke(
                {
                    "messages": agent_messages,
                },
                config=config,
                context=runtime.context,
            )

        except GraphInterrupt:
            # ------------------------------------------------------
            # HITL interrupt 必须原样向 LangGraph 传播。
            # ------------------------------------------------------
            raise

        except Exception as exc:

            error_message = str(
                exc
            ).strip()

            if not error_message:
                error_message = (
                    type(exc).__name__
                )

            failed_tool = getattr(
                exc,
                "tool_name",
                None,
            )

            # ------------------------------------------------------
            # 从已有 messages 中进一步提取 Tool Error
            # ------------------------------------------------------

            existing_messages = list(
                state.get(
                    "messages",
                    [],
                )
            )

            failure = (
                extract_tool_execution_error(
                    existing_messages
                )
            )

            if failure is not None:

                message_error = failure.get(
                    "error"
                )

                if (
                    isinstance(
                        message_error,
                        str,
                    )
                    and message_error.strip()
                ):
                    error_message = (
                        message_error
                    )

                if not failed_tool:
                    failed_tool = failure.get(
                        "last_failed_tool"
                    )

            previous_tool_results = list(
                state.get(
                    "tool_results",
                    [],
                )
            )

            return {
                "messages": state.get(
                    "messages",
                    [],
                ),
                "current_agent": (
                    "ticket_agent"
                ),
                "task_status": "failed",
                "error": error_message,
                "last_failed_node": (
                    "ticket_agent"
                ),
                "last_failed_tool": failed_tool,
                "tool_results": previous_tool_results,
            }

        # ==========================================================
        # 11. Successful Agent Invocation
        # ==========================================================

        result_messages = result.get(
            "messages",
            [],
        )

        # ==========================================================
        # 12. Extract New Messages
        # ==========================================================

        new_messages = result_messages[
            len(agent_messages):
        ]

        # ==========================================================
        # 13. Extract Tool Execution Errors
        # ==========================================================

        failure = (
            extract_tool_execution_error(
                new_messages
            )
        )

        # ==========================================================
        # 14. Collect Current Tool Results
        # ==========================================================

        current_tool_results = (
            collect_tool_results(
                result_messages
            )
        )

        # ==========================================================
        # 15. Merge Historical + Current Results
        # ==========================================================

        tool_results = _merge_tool_results(
            state.get(
                "tool_results",
                [],
            ),
            current_tool_results,
        )

        # ==========================================================
        # 16. Tool Error
        # ==============================================================

        if failure is not None:

            return {
                "messages": result_messages,
                "current_agent": (
                    "ticket_agent"
                ),
                "task_status": "failed",
                "error": failure[
                    "error"
                ],
                "last_failed_node": (
                    "ticket_agent"
                ),
                "last_failed_tool": (
                    failure[
                        "last_failed_tool"
                    ]
                ),
                "tool_results": tool_results,
            }

        # ==========================================================
        # 17. Verify Required Tool Execution
        #
        # 这是本次修改的关键。
        #
        # Ticket Agent 不能：
        #
        #     LLM 正常返回
        #         ↓
        #     task_status = completed
        #
        # 而必须：
        #
        #     required tool
        #         ↓
        #     actually executed
        #         ↓
        #     execution successful
        #         ↓
        #     completed
        # ==========================================================

        required_tools = (
            _get_required_ticket_tools(
                handoff_task
            )
        )

        if required_tools:

            missing_tools = (
                _find_missing_successful_tools(
                    required_tools,
                    current_tool_results,
                )
            )

            if missing_tools:

                missing_tool_names = ", ".join(
                    sorted(
                        missing_tools
                    )
                )

                return {
                    "messages": result_messages,
                    "current_agent": (
                        "ticket_agent"
                    ),
                    "task_status": "failed",
                    "error": (
                        "Ticket Specialist did not execute "
                        "the required tool(s): "
                        f"{missing_tool_names}"
                    ),
                    "last_failed_node": (
                        "ticket_agent"
                    ),
                    "last_failed_tool": (
                        next(
                            iter(
                                missing_tools
                            ),
                            None,
                        )
                    ),
                    "tool_results": tool_results,
                }

        # ==========================================================
        # 18. Success
        # ==========================================================

        return {
            "messages": result_messages,
            "current_agent": (
                "ticket_agent"
            ),
            "task_status": "completed",
            "tool_results": tool_results,
        }

    return ticket_agent_node


# ==============================================================
# Tool Result Merge
# ==============================================================

def _merge_tool_results(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    合并 Workflow 历史 Tool Results 与当前 invocation
    产生的 Tool Results。

    以 tool_call_id 去重。
    """

    merged: list[dict[str, Any]] = []

    seen_ids: set[str] = set()

    for result in [
        *previous,
        *current,
    ]:

        if not isinstance(
            result,
            dict,
        ):
            continue

        tool_call_id = result.get(
            "tool_call_id"
        )

        if isinstance(
            tool_call_id,
            str,
        ) and tool_call_id:

            if tool_call_id in seen_ids:
                continue

            seen_ids.add(
                tool_call_id
            )

        merged.append(
            result
        )

    return merged


# ==============================================================
# Required Ticket Tool Detection
# ==============================================================

def _get_required_ticket_tools(
    handoff_task: str | None,
) -> set[str]:
    """
    根据 Supervisor handoff_task 判断当前 Specialist
    至少需要完成哪些 Ticket Tool 操作。

    注意：
        这不是用来决定“允许哪些工具”。

    Tool permission / exposure 仍然由：
        ToolRegistry
        PermissionPolicy
        DynamicToolMiddleware

    这里仅用于：
        Task Completion Verification
    """

    if not handoff_task:
        return set()

    task = handoff_task.strip().lower()

    required_tools: set[str] = set()

    # ----------------------------------------------------------
    # Create Ticket
    # ----------------------------------------------------------

    create_keywords = (
        "create_ticket",
        "create ticket",
        "create a ticket",
        "create an incident",
        "create incident",
        "创建工单",
        "创建一个工单",
        "创建事件",
        "创建 incident",
        "创建p1",
        "创建 p1",
        "创建p2",
        "创建 p2",
        "创建p3",
        "创建 p3",
        "创建p4",
        "创建 p4",
    )

    if any(
        keyword in task
        for keyword in create_keywords
    ):
        required_tools.add(
            "create_ticket"
        )

    # ----------------------------------------------------------
    # Update Ticket
    # ----------------------------------------------------------

    update_keywords = (
        "update_ticket",
        "update ticket",
        "update the ticket",
        "modify the ticket",
        "change the ticket",
        "更新工单",
        "修改工单",
        "更新 incident",
        "修改 incident",
    )

    if any(
        keyword in task
        for keyword in update_keywords
    ):
        required_tools.add(
            "update_ticket"
        )

    # ----------------------------------------------------------
    # Get Ticket
    # ----------------------------------------------------------

    get_keywords = (
        "get_ticket",
        "get ticket",
        "query ticket",
        "query the ticket",
        "check ticket",
        "check the ticket",
        "查询工单",
        "查询 incident",
        "查看工单",
        "查看 incident",
    )

    if any(
        keyword in task
        for keyword in get_keywords
    ):
        required_tools.add(
            "get_ticket"
        )

    return required_tools


# ==============================================================
# Required Tool Success Verification
# ==============================================================

def _find_missing_successful_tools(
    required_tools: set[str],
    current_tool_results: list[dict[str, Any]],
) -> set[str]:
    """
    检查当前 Specialist invocation 是否成功执行了
    Supervisor handoff_task 所要求的工具。

    只有：
        error=False
    才算成功执行。

    注意：
        这里只检查当前 invocation 的 Tool Results，
        不使用历史 tool_results。

    原因：
        当前 handoff_task 是一个新的 Specialist 子任务，
        必须由当前 Specialist 实际完成。
    """

    successful_tools: set[str] = set()

    for result in current_tool_results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        tool_name = result.get(
            "tool_name"
        )

        if not isinstance(
            tool_name,
            str,
        ):
            continue

        if result.get(
            "error",
            False,
        ):
            continue

        successful_tools.add(
            tool_name
        )

    return (
        required_tools
        - successful_tools
    )


# ==============================================================
# Supervisor Handoff Context
# ==============================================================

def _build_handoff_context(
    handoff_task: str | None,
) -> str:
    """
    构造 Supervisor → Ticket Agent 的任务交接上下文。

    该消息被放在当前 Specialist 输入的最后，
    强化当前 handoff task 的优先级。
    """

    if not handoff_task:
        return ""

    return (
        "SUPERVISOR HANDOFF TASK\n"
        "=======================\n\n"
        "You are now executing the following task delegated "
        "by the Supervisor:\n\n"
        f"{handoff_task}\n\n"
        "Execution requirements:\n"
        "1. Complete the delegated task, not merely describe it.\n"
        "2. If the task requires creating a ticket or incident, "
        "you MUST call create_ticket.\n"
        "3. If the task requires updating a ticket, "
        "you MUST call update_ticket.\n"
        "4. If the task requires querying a ticket, "
        "you MUST call get_ticket.\n"
        "5. Do not claim the operation succeeded unless the "
        "corresponding tool actually returned success.\n"
        "6. After the required operation succeeds, stop and "
        "return a concise summary of the actual result."
    )


# ==============================================================
# Long-term Memory Context
# ==============================================================

def _build_memory_context(
    memories: list[str],
) -> str:
    """
    构造长期记忆上下文。

    Memory 只提供历史背景。

    当前 Ticket 状态必须优先依赖：
        get_ticket

    当前 create/update 是否成功必须依赖：
        当前 Tool Result
    """

    if not memories:
        return ""

    memory_text = "\n".join(
        f"- {memory}"
        for memory in memories
    )

    return (
        "RELEVANT LONG-TERM MEMORY\n"
        "=========================\n\n"
        f"{memory_text}\n\n"
        "Use these memories only as historical context. "
        "Do not treat long-term memory as authoritative for "
        "the current ticket state or the current execution "
        "result. Use live ticket tools for current state."
    )

