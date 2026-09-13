from __future__ import annotations

from typing import Any, Union

from langchain_core.tools import StructuredTool
from mcp import ClientSession
from pydantic import Field, create_model


# ============================================================
# JSON Schema -> Python Type
# ============================================================

def json_schema_to_python_type(
    schema: dict[str, Any],
) -> Any:
    """
    将 JSON Schema 字段转换为 Python 类型。

    支持：
    - string
    - integer
    - number
    - boolean
    - object
    - array
    - anyOf
    - nullable
    """

    # --------------------------------------------------------
    # 1. anyOf
    # --------------------------------------------------------
    any_of = schema.get("anyOf")

    if any_of:
        types: list[Any] = []

        for sub_schema in any_of:
            # null 分支
            if sub_schema.get("type") == "null":
                continue

            types.append(
                json_schema_to_python_type(sub_schema)
            )

        if not types:
            return Any

        if len(types) == 1:
            return types[0]

        return Union[tuple(types)]

    # --------------------------------------------------------
    # 2. oneOf
    # --------------------------------------------------------
    one_of = schema.get("oneOf")

    if one_of:
        types = [
            json_schema_to_python_type(sub_schema)
            for sub_schema in one_of
            if sub_schema.get("type") != "null"
        ]

        if not types:
            return Any

        if len(types) == 1:
            return types[0]

        return Union[tuple(types)]

    # --------------------------------------------------------
    # 3. 基础 type
    # --------------------------------------------------------
    json_type = schema.get("type")

    mapping: dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict[str, Any],
    }

    if json_type in mapping:
        return mapping[json_type]

    # --------------------------------------------------------
    # 4. array
    # --------------------------------------------------------
    if json_type == "array":
        items_schema = schema.get("items")

        if isinstance(items_schema, dict):
            item_type = json_schema_to_python_type(
                items_schema
            )
            return list[item_type]

        return list[Any]

    # --------------------------------------------------------
    # 5. 未知类型
    # --------------------------------------------------------
    return Any


# ============================================================
# JSON Schema -> Pydantic Model
# ============================================================

def json_schema_to_pydantic(
    name: str,
    schema: dict[str, Any],
):
    """
    将 MCP Tool 的 JSON Schema 转换为 Pydantic Model。

    关键行为：

    1. required 参数：
       - 必须提供
       - 使用 ...

    2. optional 参数：
       - 如果 JSON Schema 提供 default，则使用该 default
       - 否则使用 None

    3. nullable 参数：
       - 类型转换会考虑 anyOf / null

    这样可以正确处理：

        max_results: int = 5

    而不会因为 LangChain 传入 None，
    导致 MCP Server 收到：

        max_results=None
    """

    properties = schema.get(
        "properties",
        {},
    )

    required = set(
        schema.get(
            "required",
            [],
        )
    )

    fields: dict[
        str,
        tuple[Any, Any],
    ] = {}

    for field_name, field_schema in properties.items():

        if not isinstance(field_schema, dict):
            field_schema = {}

        # ----------------------------------------------------
        # 类型
        # ----------------------------------------------------
        field_type = json_schema_to_python_type(
            field_schema
        )

        description = field_schema.get(
            "description"
        )

        # ----------------------------------------------------
        # required
        # ----------------------------------------------------
        if field_name in required:
            default = ...

        # ----------------------------------------------------
        # optional
        # ----------------------------------------------------
        else:

            # MCP Schema 自己声明的 default
            if "default" in field_schema:
                default = field_schema["default"]

            # 没有 default
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


# ============================================================
# MCP Error Extraction
# ============================================================

def _extract_mcp_error_message(
    result: Any,
    tool_name: str,
) -> str:
    """
    从 MCP CallToolResult 中提取更友好的错误信息。
    """

    content = getattr(
        result,
        "content",
        None,
    )

    if content:
        messages: list[str] = []

        for item in content:

            text = getattr(
                item,
                "text",
                None,
            )

            if (
                isinstance(text, str)
                and text.strip()
            ):
                messages.append(
                    text.strip()
                )

        if messages:
            return (
                f"MCP tool '{tool_name}' failed: "
                f"{'; '.join(messages)}"
            )

    return (
        f"MCP tool '{tool_name}' execution failed"
    )


# ============================================================
# Optional Argument Normalization
# ============================================================

def _normalize_mcp_arguments(
    tool_schema: dict[str, Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    根据 MCP Tool 的 JSON Schema 清洗调用参数。

    核心目的：

    LangChain/Pydantic 可能生成：

        {
            "query": "...",
            "max_results": None,
        }

    但 MCP Tool：

        max_results: int = 5

    此时不能把 None 发送给 MCP Server。

    因此：

    - required 参数：
        None 保留，让 MCP/Pydantic 显式报错

    - optional 参数：
        None -> 删除
        让 MCP Server 使用自己的默认值

    例如：

        {
            "query": "...",
            "max_results": None,
        }

    会变成：

        {
            "query": "..."
        }
    """

    properties = tool_schema.get(
        "properties",
        {}
    )

    required = set(
        tool_schema.get(
            "required",
            []
        )
    )

    normalized: dict[str, Any] = {}

    for key, value in kwargs.items():

        field_schema = properties.get(
            key,
            {},
        )

        # ----------------------------------------------------
        # required 参数
        # ----------------------------------------------------
        if key in required:

            normalized[key] = value
            continue

        # ----------------------------------------------------
        # optional 参数为 None
        # ----------------------------------------------------
        if value is None:

            # 不把 None 发送给 MCP
            #
            # 这样：
            #
            # max_results=None
            #
            # 会变成：
            #
            # 参数不存在
            #
            # 从而让 MCP server 使用：
            #
            # max_results=5
            #
            continue

        normalized[key] = value

    return normalized


# ============================================================
# MCP Tool Call
# ============================================================

async def _call_mcp_tool(
    session: ClientSession,
    tool_name: str,
    kwargs: dict[str, Any],
    tool_schema: dict[str, Any],
):
    """
    调用 MCP Tool，并统一处理：

    1. Optional 参数清洗
    2. MCP is_error
    3. Exception propagation
    """

    # --------------------------------------------------------
    # 参数标准化
    # --------------------------------------------------------
    normalized_kwargs = _normalize_mcp_arguments(
        tool_schema=tool_schema,
        kwargs=kwargs,
    )

    # 调试日志
    print(
        f"[MCP Adapter] tool={tool_name} "
        f"arguments={normalized_kwargs}"
    )

    # --------------------------------------------------------
    # MCP Tool Call
    # --------------------------------------------------------
    result = await session.call_tool(
        tool_name,
        arguments=normalized_kwargs,
    )

    # --------------------------------------------------------
    # MCP Tool 执行失败
    # --------------------------------------------------------
    if getattr(
        result,
        "is_error",
        False,
    ):
        error_message = _extract_mcp_error_message(
            result=result,
            tool_name=tool_name,
        )

        raise RuntimeError(
            error_message
        )

    return result


# ============================================================
# MCP -> LangChain StructuredTool
# ============================================================

def convert_mcp_tool(
    session: ClientSession,
    mcp_tool: Any,
) -> StructuredTool:
    """
    将 MCP Tool 转换为 LangChain StructuredTool。
    """

    input_schema: dict[str, Any] = (
        mcp_tool.input_schema
        or {}
    )

    # --------------------------------------------------------
    # Pydantic args schema
    # --------------------------------------------------------
    args_schema = json_schema_to_pydantic(
        name=f"{mcp_tool.name}Input",
        schema=input_schema,
    )

    # --------------------------------------------------------
    # LangChain Tool Coroutine
    # --------------------------------------------------------
    async def call_mcp_tool(
        **kwargs: Any,
    ):
        return await _call_mcp_tool(
            session=session,
            tool_name=mcp_tool.name,
            kwargs=kwargs,
            tool_schema=input_schema,
        )

    # --------------------------------------------------------
    # StructuredTool
    # --------------------------------------------------------
    return StructuredTool.from_function(
        coroutine=call_mcp_tool,
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        args_schema=args_schema,
    )


# ============================================================
# Batch Conversion
# ============================================================

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