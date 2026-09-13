from typing import Any

from langchain_core.tools import StructuredTool
from mcp import ClientSession
from pydantic import Field, create_model


def json_schema_to_pydantic(
    name: str,
    schema: dict[str, Any],
):
    """
    将 MCP Tool 的 JSON Schema 转换为 Pydantic Model。
    """
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    fields: dict[str, tuple[Any, Any]] = {}

    for field_name, field_schema in properties.items():
        field_type = json_type_to_python(
            field_schema.get("type")
        )

        description = field_schema.get("description")

        if field_name in required:
            default = ...
        else:
            default = None

        fields[field_name] = (
            field_type,
            Field(
                default,
                description=description,
            ),
        )

    return create_model(
        name,
        **fields,
    )


def json_type_to_python(
    json_type: str | None,
):
    """
    JSON Schema 类型 -> Python 类型。
    """
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    return mapping.get(
        json_type,
        Any,
    )


def _extract_mcp_error_message(
    result: Any,
    tool_name: str,
) -> str:
    """
    从 MCP CallToolResult 中提取更友好的错误信息。
    """
    # 优先读取 content 中的文本错误
    content = getattr(result, "content", None)

    if content:
        messages: list[str] = []

        for item in content:
            text = getattr(item, "text", None)

            if isinstance(text, str) and text.strip():
                messages.append(text.strip())

        if messages:
            return f"MCP tool '{tool_name}' failed: {'; '.join(messages)}"

    # 如果没有可读文本，则使用通用错误
    return f"MCP tool '{tool_name}' execution failed"


async def _call_mcp_tool(
    session: ClientSession,
    tool_name: str,
    kwargs: dict[str, Any],
):
    """
    调用 MCP Tool，并统一处理 MCP 错误。

    重要：
    MCP 的 CallToolResult.is_error=True 不会自动变成 Python Exception。
    因此这里必须显式 raise，才能让 LangChain 的 Tool Retry /
    Tool Error Middleware / Recovery 链路正常工作。
    """
    result = await session.call_tool(
        tool_name,
        arguments=kwargs,
    )

    # ============================================================
    # MCP Tool 执行失败
    # ============================================================
    if getattr(result, "is_error", False):
        error_message = _extract_mcp_error_message(
            result,
            tool_name,
        )

        raise RuntimeError(error_message)

    return result


def convert_mcp_tool(
    session: ClientSession,
    mcp_tool: Any,
) -> StructuredTool:
    """
    将 MCP Tool 转换为 LangChain StructuredTool。
    """

    input_schema = mcp_tool.input_schema

    args_schema = json_schema_to_pydantic(
        name=f"{mcp_tool.name}Input",
        schema=input_schema,
    )

    async def call_mcp_tool(**kwargs):
        return await _call_mcp_tool(
            session=session,
            tool_name=mcp_tool.name,
            kwargs=kwargs,
        )

    return StructuredTool.from_function(
        coroutine=call_mcp_tool,
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        args_schema=args_schema,
    )


def convert_mcp_tools(
    session: ClientSession,
    mcp_tools: list[Any],
) -> list[StructuredTool]:
    """
    将 MCP Tools 批量转换为 LangChain StructuredTool。
    """
    return [
        convert_mcp_tool(
            session=session,
            mcp_tool=mcp_tool,
        )
        for mcp_tool in mcp_tools
    ]

