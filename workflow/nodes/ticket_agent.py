from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
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


TICKET_TOOL_SCOPE = frozenset(
    {
        "get_ticket",
        "create_ticket",
        "update_ticket",
    }
)


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

        messages = state.get(
            "messages",
            [],
        )

        retrieved_memories = state.get(
            "retrieved_memories",
            [],
        )

        memory_context = _build_memory_context(
            retrieved_memories
        )

        agent_messages = list(
            messages
        )

        if memory_context:

            agent_messages.insert(
                0,
                HumanMessage(
                    content=memory_context
                ),
            )

        # ==========================================================
        # Agent Runtime
        # ==========================================================

        try:

            result = await agent.ainvoke(
                {
                    "messages": agent_messages,
                },
                config=config,
                context=runtime.context,
            )

        # ==========================================================
        # HITL Interrupt
        #
        # 必须直接向 LangGraph 传播。
        # ==========================================================

        except GraphInterrupt:

            raise

        # ==========================================================
        # Tool Retry Exhausted / Agent Failure
        # ==========================================================

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
            # 从已有 State Message 恢复 Tool Error
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

                    failed_tool = (
                        failure.get(
                            "last_failed_tool"
                        )
                    )

            # ------------------------------------------------------
            # 保留已有 Tool Results
            # ------------------------------------------------------

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
        # Successful Agent Invocation
        # ==========================================================

        result_messages = result.get(
            "messages",
            [],
        )

        # ----------------------------------------------------------
        # 只检查本次 invocation 新增消息
        # ----------------------------------------------------------

        new_messages = result_messages[
            len(agent_messages):
        ]

        failure = (
            extract_tool_execution_error(
                new_messages
            )
        )

        # ----------------------------------------------------------
        # 当前 invocation Tool Results
        # ----------------------------------------------------------

        current_tool_results = (
            collect_tool_results(
                result_messages
            )
        )

        # ----------------------------------------------------------
        # 合并历史 Tool Results
        # ----------------------------------------------------------

        tool_results = _merge_tool_results(
            state.get(
                "tool_results",
                [],
            ),
            current_tool_results,
        )

        # ----------------------------------------------------------
        # Failure
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Success
        # ----------------------------------------------------------

        return {
            "messages": result_messages,
            "current_agent": (
                "ticket_agent"
            ),
            "task_status": "completed",
            "tool_results": tool_results,
        }

    return ticket_agent_node


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


def _build_memory_context(
    memories: list[str],
) -> str:
    """
    构造长期记忆上下文。

    Memory 只提供历史背景。

    当前 Ticket 状态：
        get_ticket
    是权威来源。

    create/update：
        必须依据当前任务和当前 Tool Result。
    """

    if not memories:

        return ""

    memory_text = "\n".join(
        f"- {memory}"
        for memory in memories
    )

    return (
        "Relevant long-term memories retrieved "
        "for this workflow:\n"
        f"{memory_text}\n\n"
        "Use these memories only as historical context. "
        "Before modifying an existing ticket, verify "
        "its current state using the appropriate ticketing "
        "tool. Do not treat long-term memory as authoritative "
        "for the current ticket state."
    )

