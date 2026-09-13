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
# Knowledge Tool Scope
# ==============================================================

KNOWLEDGE_TOOL_SCOPE = frozenset(
    {
        "rag_search",
    }
)


# ==============================================================
# Knowledge Agent System Prompt
# ==============================================================

KNOWLEDGE_AGENT_SYSTEM_PROMPT = """
You are the Knowledge Specialist Agent in an enterprise
multi-agent workflow.

Your responsibility is to execute the enterprise knowledge
retrieval task delegated by the Supervisor.

Available tool:

- rag_search

IMPORTANT EXECUTION RULES:

1. You MUST follow the Supervisor's current handoff task.

2. If the handoff task requires retrieving enterprise
   documents, internal policies, internal product
   information, or internal knowledge, you MUST call
   rag_search.

3. Do NOT claim that enterprise information has been
   retrieved unless rag_search was actually executed
   successfully.

4. Base conclusions on the retrieved enterprise knowledge,
   not on unsupported assumptions.

5. Long-term memory is historical context and must not
   replace current enterprise retrieval.

6. Do not perform unrelated operations.

7. When the required retrieval succeeds, summarize the
   actual retrieved information and stop.

8. A natural-language response alone does NOT mean the
   delegated knowledge task has been completed.

9. The surrounding Agent Runtime is responsible for:
   Permission Policy, Retry, Error Handling, Recovery,
   and Human-in-the-Loop.

10. Treat the Supervisor handoff task as the current
    execution boundary for this Specialist.
""".strip()


# ==============================================================
# Create Knowledge Agent
# ==============================================================

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
                    KNOWLEDGE_AGENT_SYSTEM_PROMPT
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
                    "knowledge_agent"
                ),
                "task_status": "failed",
                "error": error_message,
                "last_failed_node": (
                    "knowledge_agent"
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
        # 13. Required Tool Verification
        # ==========================================================

        required_tools = (
            _get_required_knowledge_tools(
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
                        "knowledge_agent"
                    ),
                    "task_status": "failed",
                    "error": (
                        "Knowledge Specialist did not execute "
                        "the required tool(s): "
                        f"{missing_tool_names}"
                    ),
                    "last_failed_node": (
                        "knowledge_agent"
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
                "knowledge_agent"
            ),
            "task_status": "completed",
            "tool_results": tool_results,
        }

    return knowledge_agent_node


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
# Required Knowledge Tool Detection
# ==============================================================

def _get_required_knowledge_tools(
    handoff_task: str | None,
) -> set[str]:
    """
    根据 Supervisor handoff_task 判断是否要求
    Knowledge Specialist 执行 rag_search。
    """

    if not handoff_task:
        return set()

    task = handoff_task.strip().lower()

    required_tools: set[str] = set()

    knowledge_keywords = (
        "rag_search",
        "rag search",
        "search the internal",
        "search internal",
        "retrieve internal",
        "retrieve enterprise",
        "enterprise knowledge",
        "internal knowledge",
        "internal document",
        "internal documents",
        "internal policy",
        "internal policies",
        "product documentation",
        "enterprise policy",
        "企业知识",
        "企业内部知识",
        "内部知识",
        "内部文档",
        "企业文档",
        "内部政策",
        "企业政策",
        "产品文档",
        "知识库",
        "检索知识",
        "检索文档",
    )

    if any(
        keyword in task
        for keyword in knowledge_keywords
    ):
        required_tools.add(
            "rag_search"
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
    检查当前 Specialist invocation 是否真正成功执行
    Supervisor 所要求的 Tool。
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
    构造 Supervisor → Knowledge Agent 的任务交接上下文。
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
        "2. If the task requires enterprise internal knowledge "
        "retrieval, documents, policies, or internal information, "
        "you MUST call rag_search.\n"
        "3. Do not claim that enterprise information has been "
        "retrieved unless rag_search actually returned results.\n"
        "4. Base conclusions on retrieved enterprise knowledge "
        "rather than unsupported assumptions.\n"
        "5. After the required retrieval succeeds, stop and "
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

    当前 enterprise knowledge / document facts
    必须优先来自 rag_search。
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
        "For enterprise internal knowledge, documents, policies, "
        "or product information, always rely on current "
        "rag_search results."
    )

