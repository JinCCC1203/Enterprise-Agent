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
# Operations Tool Scope
# ==============================================================

OPERATIONS_TOOL_SCOPE = frozenset(
    {
        "get_service_health",
    }
)


# ==============================================================
# Operations Agent System Prompt
# ==============================================================

OPERATIONS_AGENT_SYSTEM_PROMPT = """
You are the Operations Specialist Agent in an enterprise
multi-agent workflow.

Your responsibility is to execute the operational task
delegated by the Supervisor.

Available tool:

- get_service_health

IMPORTANT EXECUTION RULES:

1. You MUST follow the Supervisor's current handoff task.

2. If the handoff task requires checking service health,
   service status, latency, error rate, or operational
   state, you MUST call get_service_health.

3. Do NOT claim that a service has been checked unless
   get_service_health was actually executed successfully.

4. Current operational state must come from the live
   get_service_health result rather than historical memory.

5. Do not perform unrelated operations.

6. When the required tool operation succeeds, summarize
   the actual tool result and stop.

7. A natural-language response alone does NOT mean that
   the delegated operational task has been completed.

8. The surrounding Agent Runtime is responsible for:
   Permission Policy, Retry, Error Handling, Recovery,
   and Human-in-the-Loop.

9. Treat the Supervisor handoff task as the current
   execution boundary for this Specialist.
""".strip()


# ==============================================================
# Create Operations Agent
# ==============================================================

def create_operations_agent(
    *,
    model: ChatOpenAI,
    registry: ToolRegistry,
    permission_policy: PermissionPolicy,
    middleware: list[Any] | None = None,
):
    """
    创建 Operations Specialist Agent。

    Scope:
        get_service_health
    """

    # ==============================================================
    # 1. Specialist Registry View
    # ==============================================================

    registry_view = registry.create_view(
        OPERATIONS_TOOL_SCOPE
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
        agent_name="operations_agent",
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

    async def operations_agent_node(
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
        # 9. Build Agent Messages
        # ==========================================================

        agent_messages: list[Any] = []

        agent_messages.append(
            SystemMessage(
                content=(
                    OPERATIONS_AGENT_SYSTEM_PROMPT
                )
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
                    failed_tool = (
                        failure.get(
                            "last_failed_tool"
                        )
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
                    "operations_agent"
                ),
                "task_status": "failed",
                "error": error_message,
                "last_failed_node": (
                    "operations_agent"
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

        new_messages = result_messages[
            len(agent_messages):
        ]

        failure = (
            extract_tool_execution_error(
                new_messages
            )
        )

        current_tool_results = (
            collect_tool_results(
                result_messages
            )
        )

        tool_results = _merge_tool_results(
            state.get(
                "tool_results",
                [],
            ),
            current_tool_results,
        )

        # ==========================================================
        # 12. Tool Error
        # ==========================================================

        if failure is not None:

            return {
                "messages": result_messages,
                "current_agent": (
                    "operations_agent"
                ),
                "task_status": "failed",
                "error": failure[
                    "error"
                ],
                "last_failed_node": (
                    "operations_agent"
                ),
                "last_failed_tool": (
                    failure[
                        "last_failed_tool"
                    ]
                ),
                "tool_results": tool_results,
            }

        # ==========================================================
        # 13. Required Tool Verification
        # ==========================================================

        required_tools = (
            _get_required_operations_tools(
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

                missing_tool = next(
                    iter(
                        missing_tools
                    ),
                    None,
                )

                return {
                    "messages": result_messages,
                    "current_agent": (
                        "operations_agent"
                    ),
                    "task_status": "failed",
                    "error": (
                        "Operations Specialist did not execute "
                        "the required tool(s): "
                        f"{missing_tool_names}"
                    ),
                    "last_failed_node": (
                        "operations_agent"
                    ),
                    "last_failed_tool": missing_tool,
                    "tool_results": tool_results,
                }

        # ==========================================================
        # 14. Success
        # ==========================================================

        return {
            "messages": result_messages,
            "current_agent": (
                "operations_agent"
            ),
            "task_status": "completed",
            "tool_results": tool_results,
        }

    return operations_agent_node


# ==============================================================
# Tool Result Merge
# ==============================================================

def _merge_tool_results(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    合并历史与当前 Tool Results。

    通过 tool_call_id 去重。
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
# Required Operations Tool Detection
# ==============================================================

def _get_required_operations_tools(
    handoff_task: str | None,
) -> set[str]:
    """
    根据 Supervisor handoff_task 判断当前 Operations
    Specialist 需要执行哪些工具。

    这里只负责 Task Completion Verification，
    不负责 Tool Permission / Exposure。
    """

    if not handoff_task:
        return set()

    task = handoff_task.strip().lower()

    required_tools: set[str] = set()

    health_keywords = (
        "get_service_health",
        "service health",
        "service status",
        "health status",
        "health check",
        "service condition",
        "latency",
        "error rate",
        "服务健康",
        "服务状态",
        "健康状态",
        "健康检查",
        "检查服务",
        "查询服务",
        "运行状态",
        "延迟",
        "错误率",
        "异常状态",
    )

    if any(
        keyword in task
        for keyword in health_keywords
    ):
        required_tools.add(
            "get_service_health"
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
    检查当前 Specialist invocation 是否成功执行
    所要求的 Tool。

    只有当前 invocation 中：
        error=False

    的 Tool Result 才计为成功。

    不使用历史 tool_results 判断当前任务，
    避免旧结果错误地满足新的 handoff_task。
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
    构造 Supervisor → Operations Agent 的任务交接上下文。
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
        "2. If the task requires checking service health or "
        "operational status, you MUST call get_service_health.\n"
        "3. Do not claim the service has been checked unless "
        "the tool actually returned a result.\n"
        "4. For current service state, trust the live tool result "
        "over historical memory.\n"
        "5. After the required operation succeeds, stop and "
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

    历史 Memory 只提供背景信息。
    当前服务状态必须以 get_service_health 为准。
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
        "For current service health, status, latency, or "
        "error rate, always rely on the live "
        "get_service_health result."
    )

