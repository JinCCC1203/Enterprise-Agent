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


KNOWLEDGE_TOOL_SCOPE = frozenset(
    {
        "rag_search",
    }
)


def create_knowledge_agent(
    *,
    model: ChatOpenAI,
    registry: ToolRegistry,
    permission_policy: PermissionPolicy,
    middleware: list[Any] | None = None,
):
    """
    创建 Knowledge Specialist Agent。

    Scope:
        rag_search
    """

    # ==============================================================
    # 1. Specialist Registry View
    # ==============================================================

    registry_view = registry.create_view(
        KNOWLEDGE_TOOL_SCOPE
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
        agent_name="knowledge_agent",
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

    async def knowledge_agent_node(
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

        memory_context = (
            _build_memory_context(
                retrieved_memories
            )
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

                extracted_error = (
                    failure.get(
                        "error"
                    )
                )

                if (
                    isinstance(
                        extracted_error,
                        str,
                    )
                    and extracted_error.strip()
                ):

                    error_message = (
                        extracted_error
                    )

                if not failed_tool:

                    failed_tool = (
                        failure.get(
                            "last_failed_tool"
                        )
                    )

            return {
                "messages": state.get(
                    "messages",
                    [],
                ),
                "current_agent": (
                    "knowledge_agent"
                ),
                "task_status": "failed",
                "error": error_message,
                "last_failed_node": (
                    "knowledge_agent"
                ),
                "last_failed_tool": failed_tool,
                "tool_results": list(
                    state.get(
                        "tool_results",
                        [],
                    )
                ),
            }

        # ==========================================================
        # Successful Agent Invocation
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
        # Failure
        # ==========================================================

        if failure is not None:

            return {
                "messages": result_messages,
                "current_agent": (
                    "knowledge_agent"
                ),
                "task_status": "failed",
                "error": failure[
                    "error"
                ],
                "last_failed_node": (
                    "knowledge_agent"
                ),
                "last_failed_tool": (
                    failure[
                        "last_failed_tool"
                    ]
                ),
                "tool_results": tool_results,
            }

        # ==========================================================
        # Success
        # ==========================================================

        return {
            "messages": result_messages,
            "current_agent": (
                "knowledge_agent"
            ),
            "task_status": "completed",
            "tool_results": tool_results,
        }

    return knowledge_agent_node


def _merge_tool_results(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    合并历史与当前 Tool Results。
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
        "For current enterprise knowledge or document "
        "information, always rely on rag_search results "
        "over historical memory."
    )

