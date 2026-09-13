from __future__ import annotations

import ast
import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from workflow.state import EnterpriseAgentState
from workflow.utils.execution_facts import build_execution_summary


def create_finalizer_node():
    """
    创建 Workflow Finalizer。

    Finalizer 只消费 Runtime State 和 Tool Execution Facts，
    不调用 Tool、不修改业务数据、不参与 Routing。
    """

    async def finalizer_node(
        state: EnterpriseAgentState,
    ) -> dict[str, Any]:
        execution_summary = build_execution_summary(state)

        task_status = execution_summary.get("task_status")
        tool_results = execution_summary.get("tool_results", [])

        # ==========================================================
        # Workflow Failure
        # ==========================================================

        if task_status == "failed":
            answer = _build_failure_answer(execution_summary)

            return {
                "execution_summary": execution_summary,
                "final_answer": answer,
            }

        # ==========================================================
        # Tool-based Workflow
        # ==========================================================

        if tool_results:
            answer = _build_tool_answer(
                execution_summary=execution_summary,
            )

            return {
                "execution_summary": execution_summary,
                "final_answer": answer,
            }

        # ==========================================================
        # Normal Agent Conversation
        # ==========================================================

        answer = _extract_latest_ai_message(
            state.get("messages", [])
        )

        if not answer:
            answer = (
                "Workflow completed successfully, "
                "but no final answer was generated."
            )

        return {
            "execution_summary": execution_summary,
            "final_answer": answer,
        }

    return finalizer_node


# ==============================================================
# Tool Execution Answer
# ==============================================================

def _build_tool_answer(
    *,
    execution_summary: dict[str, Any],
) -> str:
    """
    构造 Tool 执行类任务的最终答案。

    所有关键事实来自 Runtime State / Execution Facts。
    """

    lines: list[str] = []

    task_status = execution_summary.get(
        "task_status"
    )

    workflow_complete = bool(
        execution_summary.get(
            "workflow_complete",
            False,
        )
    )

    completed_agents = execution_summary.get(
        "completed_agents",
        [],
    )

    # ==========================================================
    # Workflow Status
    # ==========================================================

    if (
        task_status == "completed"
        and workflow_complete
    ):
        lines.append(
            "✅ Workflow 已成功完成。"
        )

    elif task_status == "completed":
        lines.append(
            "Workflow 已执行完成当前阶段。"
        )

    else:
        lines.append(
            f"Workflow 状态：{task_status}"
        )

    # ==========================================================
    # Completed Specialists
    # ==========================================================

    normalized_agents = [
        str(agent)
        for agent in completed_agents
        if agent
    ]

    if normalized_agents:
        lines.append(
            "已完成的 Specialist："
            + ", ".join(normalized_agents)
        )

    # ==========================================================
    # HITL
    # ==========================================================

    _append_approval_facts(
        lines=lines,
        execution_summary=execution_summary,
    )

    # ==========================================================
    # Recovery
    # ==========================================================

    _append_recovery_facts(
        lines=lines,
        execution_summary=execution_summary,
    )

    # ==========================================================
    # Tool Results
    # ==========================================================

    tool_results = execution_summary.get(
        "tool_results",
        [],
    )

    _append_tool_results(
        lines=lines,
        execution_summary=execution_summary,
    )

    # ==========================================================
    # Business Summary
    # ==========================================================

    _append_business_summary(
        lines=lines,
        tool_results=tool_results,
    )

    return "\n".join(lines)


# ==============================================================
# Approval Facts
# ==============================================================

def _append_approval_facts(
    *,
    lines: list[str],
    execution_summary: dict[str, Any],
) -> None:
    """
    输出 Tool-level HITL 事实。
    """

    approval_required = bool(
        execution_summary.get(
            "approval_required",
            False,
        )
    )

    approval_status = execution_summary.get(
        "approval_status"
    )

    approval_events = execution_summary.get(
        "approval_events",
        [],
    )

    if approval_required:
        lines.append(
            "人工审批：当前仍在等待审批。"
        )

        if approval_status:
            lines.append(
                f"当前审批状态：{approval_status}"
            )

        return

    if not approval_events:
        return

    latest_event = approval_events[-1]

    if not isinstance(
        latest_event,
        dict,
    ):
        return

    decision = latest_event.get(
        "decision"
    )

    tool_names = latest_event.get(
        "tool_names",
        [],
    )

    if not isinstance(
        tool_names,
        list,
    ):
        tool_names = []

    if decision == "approved":
        lines.append(
            "最近一次 Tool 审批：已通过。"
        )

    elif decision == "edited":
        lines.append(
            "最近一次 Tool 审批：已通过修改后的请求。"
        )

    elif decision == "rejected":
        lines.append(
            "最近一次 Tool 审批：已拒绝。"
        )

    elif decision:
        lines.append(
            f"最近一次 Tool 审批：{decision}"
        )

    normalized_names = [
        str(name)
        for name in tool_names
        if name
    ]

    if normalized_names:
        lines.append(
            "审批工具："
            + ", ".join(normalized_names)
        )

    if (
        approval_status
        and decision
        and approval_status != decision
    ):
        lines.append(
            "注意：当前审批状态与最近审批事件存在不一致，"
            "以审批事件历史为准。"
        )


# ==============================================================
# Recovery Facts
# ==============================================================

def _append_recovery_facts(
    *,
    lines: list[str],
    execution_summary: dict[str, Any],
) -> None:
    """
    输出 Workflow Recovery 信息。
    """

    recovery_status = execution_summary.get(
        "recovery_status"
    )

    recovery_attempts = execution_summary.get(
        "recovery_attempts",
        0,
    )

    recovery_reason = execution_summary.get(
        "recovery_reason"
    )

    if not recovery_status:
        return

    lines.append(
        f"\nWorkflow Recovery：{recovery_status}"
    )

    if recovery_attempts:
        lines.append(
            f"Recovery Attempts：{recovery_attempts}"
        )

    if recovery_reason:
        lines.append(
            f"Recovery Reason：{recovery_reason}"
        )


# ==============================================================
# Tool Results
# ==============================================================

def _append_tool_results(
    *,
    lines: list[str],
    execution_summary: dict[str, Any],
) -> None:
    """
    输出 Tool Execution Facts。

    不直接输出 MCP CallToolResult 对象。
    """

    tool_results = execution_summary.get(
        "tool_results",
        [],
    )

    if not tool_results:
        return

    successful_tools: list[dict[str, Any]] = []
    failed_tools: list[dict[str, Any]] = []

    for result in tool_results:
        if not isinstance(
            result,
            dict,
        ):
            continue

        if result.get(
            "error",
            False,
        ):
            failed_tools.append(result)
        else:
            successful_tools.append(result)

    if successful_tools:
        lines.append(
            "\n已成功执行的工具："
        )

        for result in successful_tools:
            tool_name = result.get(
                "tool_name",
                "unknown",
            )

            lines.append(
                f"- {tool_name}"
            )

            lines.append(
                _format_tool_content(
                    result.get("content")
                )
            )

    if failed_tools:
        lines.append(
            "\n工具执行失败："
        )

        for result in failed_tools:
            tool_name = result.get(
                "tool_name",
                "unknown",
            )

            error_type = result.get(
                "error_type"
            )

            error_message = result.get(
                "error_message"
            )

            if error_type:
                lines.append(
                    f"- {tool_name} ({error_type})"
                )
            else:
                lines.append(
                    f"- {tool_name}"
                )

            if error_message:
                lines.append(
                    f"  错误：{error_message}"
                )


# ==============================================================
# Business Summary
# ==============================================================

def _append_business_summary(
    *,
    lines: list[str],
    tool_results: list[Any],
) -> None:
    """
    根据真实 Tool Execution Facts 输出业务摘要。
    """

    health_results = _successful_tool_results(
        tool_results,
        "get_service_health",
    )

    create_results = _successful_tool_results(
        tool_results,
        "create_ticket",
    )

    update_results = _successful_tool_results(
        tool_results,
        "update_ticket",
    )

    get_ticket_results = _successful_tool_results(
        tool_results,
        "get_ticket",
    )

    # ==========================================================
    # Service Health
    # ==========================================================

    for result in health_results:
        data = _extract_structured_data(
            result.get("content")
        )

        if not isinstance(
            data,
            dict,
        ):
            continue

        health_data = data.get("data")

        if not isinstance(
            health_data,
            dict,
        ):
            health_data = data

        service = health_data.get(
            "service"
        )

        status = health_data.get(
            "status"
        )

        latency = health_data.get(
            "latency_ms"
        )

        error_rate = health_data.get(
            "error_rate"
        )

        if not service or not status:
            continue

        summary_parts = [
            f"{service} 当前状态为 {status}"
        ]

        if latency is not None:
            summary_parts.append(
                f"延迟 {latency}ms"
            )

        if error_rate is not None:
            try:
                error_rate_percent = (
                    float(error_rate)
                    * 100
                )

                summary_parts.append(
                    f"错误率 {error_rate_percent:.1f}%"
                )

            except (
                TypeError,
                ValueError,
            ):
                summary_parts.append(
                    f"错误率 {error_rate}"
                )

        lines.append(
            "服务健康检查："
            + "，".join(summary_parts)
            + "。"
        )

    # ==========================================================
    # Create Ticket
    # ==========================================================

    for result in create_results:
        data = _extract_structured_data(
            result.get("content")
        )

        if not isinstance(
            data,
            dict,
        ):
            continue

        if data.get("success") is False:
            continue

        ticket_data = data.get(
            "data"
        )

        if not isinstance(
            ticket_data,
            dict,
        ):
            ticket_data = data

        ticket_id = ticket_data.get(
            "ticket_id"
        )

        title = ticket_data.get(
            "title"
        )

        priority = ticket_data.get(
            "priority"
        )

        status = ticket_data.get(
            "status"
        )

        service_name = ticket_data.get(
            "service_name"
        )

        summary_parts = [
            "已成功创建 Incident 工单"
        ]

        if ticket_id:
            summary_parts.append(
                f"工单号 {ticket_id}"
            )

        if priority:
            summary_parts.append(
                f"优先级 {priority}"
            )

        if service_name:
            summary_parts.append(
                f"服务 {service_name}"
            )

        if title:
            summary_parts.append(
                f"标题「{title}」"
            )

        if status:
            summary_parts.append(
                f"当前状态 {status}"
            )

        lines.append(
            "工单处理："
            + "，".join(summary_parts)
            + "。"
        )

    # ==========================================================
    # Update Ticket
    # ==========================================================

    for result in update_results:
        data = _extract_structured_data(
            result.get("content")
        )

        if not isinstance(
            data,
            dict,
        ):
            continue

        ticket_data = data.get(
            "data"
        )

        if not isinstance(
            ticket_data,
            dict,
        ):
            ticket_data = data

        ticket_id = ticket_data.get(
            "ticket_id"
        )

        status = ticket_data.get(
            "status"
        )

        if ticket_id and status:
            lines.append(
                "工单更新："
                f"{ticket_id} 当前状态为 {status}。"
            )

    # ==========================================================
    # Get Ticket
    # ==========================================================

    for result in get_ticket_results:
        data = _extract_structured_data(
            result.get("content")
        )

        if not isinstance(
            data,
            dict,
        ):
            continue

        ticket_data = data.get(
            "data"
        )

        if not isinstance(
            ticket_data,
            dict,
        ):
            ticket_data = data

        ticket_id = ticket_data.get(
            "ticket_id"
        )

        status = ticket_data.get(
            "status"
        )

        priority = ticket_data.get(
            "priority"
        )

        if not ticket_id:
            continue

        summary_parts = [
            f"工单 {ticket_id}"
        ]

        if priority:
            summary_parts.append(
                f"优先级 {priority}"
            )

        if status:
            summary_parts.append(
                f"状态 {status}"
            )

        lines.append(
            "工单查询："
            + "，".join(summary_parts)
            + "。"
        )


# ==============================================================
# Successful Tool Results
# ==============================================================

def _successful_tool_results(
    tool_results: list[Any],
    tool_name: str,
) -> list[dict[str, Any]]:
    """
    获取指定 Tool 的成功执行结果。
    """

    results: list[dict[str, Any]] = []

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

        results.append(result)

    return results


# ==============================================================
# Failure Answer
# ==============================================================

def _build_failure_answer(
    execution_summary: dict[str, Any],
) -> str:
    """
    构造 Workflow Failure 最终答案。
    """

    lines = [
        "❌ Workflow 未能成功完成。"
    ]

    failed_node = execution_summary.get(
        "last_failed_node"
    )

    if failed_node:
        lines.append(
            f"失败节点：{failed_node}"
        )

    failed_tool = execution_summary.get(
        "last_failed_tool"
    )

    if failed_tool:
        lines.append(
            f"失败工具：{failed_tool}"
        )

    _append_recovery_facts(
        lines=lines,
        execution_summary=execution_summary,
    )

    error = execution_summary.get(
        "error"
    )

    if error:
        lines.append(
            f"错误原因：{error}"
        )

    _append_tool_results(
        lines=lines,
        execution_summary=execution_summary,
    )

    _append_approval_facts(
        lines=lines,
        execution_summary=execution_summary,
    )

    return "\n".join(lines)


# ==============================================================
# Normal Agent Answer
# ==============================================================

def _extract_latest_ai_message(
    messages: list[BaseMessage],
) -> str:
    """
    普通 Agent 场景下提取最后一个 AIMessage。
    """

    for message in reversed(messages):
        if not isinstance(
            message,
            AIMessage,
        ):
            continue

        content = message.content

        if isinstance(
            content,
            str,
        ):
            text = content.strip()

            if text:
                return text

        if isinstance(
            content,
            list,
        ):
            text = _extract_text_from_content_blocks(
                content
            )

            if text:
                return text

    return ""


# ==============================================================
# Tool Content Formatting
# ==============================================================

def _format_tool_content(
    content: Any,
) -> str:
    """
    将 Tool Result 转成用户可读内容。
    """

    if content is None:
        return "  执行结果：None"

    structured = _extract_structured_data(
        content
    )

    if structured is not None:
        return (
            "  执行结果："
            + _safe_json_string(
                structured
            )
        )

    text = _extract_text_content(
        content
    )

    if text:
        return (
            "  执行结果："
            f"{text}"
        )

    return (
        "  执行结果："
        f"{_one_line(content)}"
    )


# ==============================================================
# Structured Content Extraction
# ==============================================================

def _extract_structured_data(
    content: Any,
) -> Any:
    """
    从 MCP / LangChain Tool Result 中提取结构化数据。
    """

    if content is None:
        return None

    # ----------------------------------------------------------
    # Dict
    # ----------------------------------------------------------

    if isinstance(
        content,
        dict,
    ):
        if "structured_content" in content:
            return content.get(
                "structured_content"
            )

        if "structuredContent" in content:
            return content.get(
                "structuredContent"
            )

        return content

    # ----------------------------------------------------------
    # Pydantic / MCP Object
    # ----------------------------------------------------------

    model_dump = getattr(
        content,
        "model_dump",
        None,
    )

    if callable(model_dump):
        try:
            dumped = model_dump()

            if isinstance(
                dumped,
                dict,
            ):
                if "structured_content" in dumped:
                    return dumped.get(
                        "structured_content"
                    )

                if "structuredContent" in dumped:
                    return dumped.get(
                        "structuredContent"
                    )

                if dumped:
                    return dumped

        except Exception:
            pass

    # ----------------------------------------------------------
    # Direct Attribute
    # ----------------------------------------------------------

    structured_content = getattr(
        content,
        "structured_content",
        None,
    )

    if structured_content is not None:
        return structured_content

    structured_content = getattr(
        content,
        "structuredContent",
        None,
    )

    if structured_content is not None:
        return structured_content

    # ----------------------------------------------------------
    # MCP Content
    # ----------------------------------------------------------

    message_content = getattr(
        content,
        "content",
        None,
    )

    if isinstance(
        message_content,
        list,
    ):
        extracted = _extract_content_blocks(
            message_content
        )

        if extracted is not None:
            return extracted

    # ----------------------------------------------------------
    # String Representation
    # ----------------------------------------------------------

    if isinstance(
        content,
        str,
    ):
        return _extract_structured_from_string(
            content
        )

    return None


# ==============================================================
# Structured Content From String
# ==============================================================

def _extract_structured_from_string(
    text: str,
) -> Any:
    """
    从 CallToolResult 字符串中提取 structured_content。
    """

    marker = "structured_content="
    index = text.find(marker)

    if index == -1:
        marker = "structuredContent="
        index = text.find(marker)

    if index == -1:
        return None

    payload = text[
        index + len(marker):
    ].strip()

    if not payload.startswith(
        "{"
    ):
        return None

    extracted = _extract_balanced_dict(
        payload
    )

    if not extracted:
        return None

    try:
        return ast.literal_eval(
            extracted
        )
    except Exception:
        pass

    try:
        return json.loads(
            extracted
        )
    except Exception:
        return None


# ==============================================================
# Balanced Dict Extraction
# ==============================================================

def _extract_balanced_dict(
    text: str,
) -> str | None:
    """
    提取第一个完整的 Python dict / JSON object。
    """

    if not text.startswith(
        "{"
    ):
        return None

    depth = 0
    in_string = False
    quote_char = ""
    escaped = False

    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue

        if (
            in_string
            and char == "\\"
        ):
            escaped = True
            continue

        if char in {
            "'",
            '"',
        }:
            if not in_string:
                in_string = True
                quote_char = char
            elif char == quote_char:
                in_string = False

            continue

        if in_string:
            continue

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                return text[
                    : index + 1
                ]

    return None


# ==============================================================
# Generic Text Extraction
# ==============================================================

def _extract_text_content(
    content: Any,
) -> str:
    """
    提取 Tool Result 中的文本。
    """

    if content is None:
        return ""

    if isinstance(
        content,
        str,
    ):
        return content.strip()

    if isinstance(
        content,
        dict,
    ):
        text = content.get(
            "text"
        )

        if isinstance(
            text,
            str,
        ):
            return text.strip()

        return ""

    text = getattr(
        content,
        "text",
        None,
    )

    if isinstance(
        text,
        str,
    ):
        return text.strip()

    inner_content = getattr(
        content,
        "content",
        None,
    )

    if isinstance(
        inner_content,
        list,
    ):
        return _extract_text_from_content_blocks(
            inner_content
        )

    return ""


# ==============================================================
# Content Block Extraction
# ==============================================================

def _extract_content_blocks(
    blocks: list[Any],
) -> Any:
    """
    从 MCP Content Blocks 中提取结构化数据或文本。
    """

    text_parts: list[str] = []

    for block in blocks:
        if isinstance(
            block,
            dict,
        ):
            if "structured_content" in block:
                return block.get(
                    "structured_content"
                )

            if "structuredContent" in block:
                return block.get(
                    "structuredContent"
                )

            text = block.get(
                "text"
            )

            if isinstance(
                text,
                str,
            ):
                text_parts.append(text)

            continue

        structured = getattr(
            block,
            "structured_content",
            None,
        )

        if structured is not None:
            return structured

        text = getattr(
            block,
            "text",
            None,
        )

        if isinstance(
            text,
            str,
        ):
            text_parts.append(text)

    if not text_parts:
        return None

    text = "\n".join(
        text_parts
    ).strip()

    if not text:
        return None

    try:
        return json.loads(
            text
        )
    except Exception:
        return text


# ==============================================================
# Text From Content Blocks
# ==============================================================

def _extract_text_from_content_blocks(
    blocks: list[Any],
) -> str:
    """
    提取 Content Blocks 中的文本。
    """

    text_parts: list[str] = []

    for block in blocks:
        if isinstance(
            block,
            str,
        ):
            text_parts.append(block)
            continue

        if isinstance(
            block,
            dict,
        ):
            text = block.get(
                "text"
            )

            if isinstance(
                text,
                str,
            ):
                text_parts.append(text)

            continue

        text = getattr(
            block,
            "text",
            None,
        )

        if isinstance(
            text,
            str,
        ):
            text_parts.append(text)

    return "\n".join(
        text_parts
    ).strip()


# ==============================================================
# Safe JSON Formatting
# ==============================================================

def _safe_json_string(
    value: Any,
) -> str:
    """
    将结构化 Tool Result 格式化为 JSON。
    """

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(
                ", ",
                ": ",
            ),
        )
    except (
        TypeError,
        ValueError,
    ):
        return _one_line(value)


# ==============================================================
# One-line Utility
# ==============================================================

def _one_line(
    value: Any,
) -> str:
    """
    将对象转换为单行文本。
    """

    if value is None:
        return "None"

    return " ".join(
        str(value).split()
    )

