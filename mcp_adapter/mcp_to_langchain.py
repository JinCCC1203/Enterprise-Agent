from typing import Any

from pydantic import create_model,Field
from langchain_core.tools import StructuredTool

from mcp import ClientSession

def json_schema_to_pydantic(
        name: str,
        schema: dict[str,Any]
):
    """
    将 MCP tool 的 Json schema 转换为 Pydantic Model
    """
    properties=schema.get("properties",{})
    required=schema.get("required",[])

    fields: dict[str, tuple[Any, Any]] = {}

    for field_name,field_schema in properties.items():
        field_type=json_type_to_python(
            field_schema.get("type")
        )
        description=field_schema.get("description")
        if field_name in required:
            default=...
        else:
            default=None
        fields[field_name]=(
            field_type,
            Field(
                default,
                description=description
            )
        )

    return create_model(
            name,
            **fields
        )

def json_type_to_python(
        json_type: str | None
):
    """
    Json Schema 类型 -> Python类型
    """
    mapping={
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    return mapping.get(
        json_type, Any
    )

def convert_mcp_tool(
        session: ClientSession,
        mcp_tool: Any
) -> StructuredTool:
    """
    将 MCP tool 真正转换为Langchain tool
    """

    input_schema=mcp_tool.input_schema
    args_schema=json_schema_to_pydantic(
        name=f"{mcp_tool.name}Input",
        schema=input_schema
    )

    async def call_mcp_tool(**kwargs):
        result=await session.call_tool(
            mcp_tool.name,
            arguments=kwargs
        )
        return result

    return StructuredTool.from_function(
        coroutine=call_mcp_tool,
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        args_schema=args_schema
    )

def convert_mcp_tools(
        session: ClientSession,
        mcp_tools: list[Any]
) -> list[StructuredTool]:

    return [
        convert_mcp_tool(
            session,
            mcp_tool
        )
        for mcp_tool in mcp_tools
    ]