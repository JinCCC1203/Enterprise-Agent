# tools/registry.py

from __future__ import annotations

from typing import Iterable

from langchain_core.tools import BaseTool

from .metadata import ToolMetadata


class ToolRegistry:
    """
    统一管理 Agent Runtime 中的所有 Tool。

    Registry 负责：
    1. 注册 Tool
    2. 获取 Tool
    3. 获取 Tool Metadata
    4. 按条件查询 Tool
    """

    def __init__(self) -> None:
        self._tools: dict[str,BaseTool]={}
        self._metadata: dict[str,ToolMetadata]={}

    def register(
            self,
            tool: BaseTool,
            metadata: ToolMetadata
    ) -> None:
        """
        注册一个 Tool。

        Tool 的真实名字以 tool.name 为准，
        metadata.name 必须与其保持一致
        """
        if tool.name != metadata.name:
            raise ValueError(
                f"Tool name mismatch: "
                f"tool.name={tool.name!r},"
                f"metadata.name={metadata.name!r}"
            )

        if tool.name in self._tools:
            raise ValueError(
                f"Tool already registered: {tool.name}"
            )

        self._tools[tool.name]=tool
        self._metadata[tool.name]=metadata

    def register_many(
            self,
            tools: Iterable[tuple[BaseTool,ToolMetadata]]
    ) -> None:
        """
        批量注册Tool
        """
        for tool,metadata in tools:
            self.register(tool,metadata)

    def unregister(self,tool_name: str) -> None:
        """
        删除Tool
        """
        self._tools.pop(tool_name,None)
        self._metadata.pop(tool_name,None)

    def get(self,tool_name: str) -> BaseTool:
        """
        根据 Tool name 获取 Tool。
        """
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise KeyError(
                f"Tool not found: {tool_name}"
            ) from exc

    def get_metadata(self,tool_name: str) -> ToolMetadata:
        """
        根据 Tool 元数据。
        """
        try:
            return self._metadata[tool_name]
        except KeyError as exc:
            raise KeyError(
                f"Metadata not found: {tool_name}"
            ) from exc

    def get_all_tools(self) -> list[BaseTool]:
        """
        返回所有已注册 Tool。
        """
        return list(self._tools.values())

    def get_all_metadata(self) -> list[ToolMetadata]:
        """
        返回所有 Tool Metadata。
        """
        return list(self._metadata.values())

    def get_by_category(self,category: str) -> list[BaseTool]:
        """
        按 category 获取 Tool
        """
        return [
            self._tools[name]
            for name,metadata in self._metadata.items()
            if metadata.category == category
        ]

    def get_by_source(self,source: str) -> list[BaseTool]:
        """
        按来源获取 Tool
        """
        return [
            self._tools[name]
            for name,metadata in self._metadata.items()
            if metadata.source.value == source
        ]

    def get_by_permission(self,role: str) -> list[BaseTool]:
        """
        获取某个角色允许使用的 Tool
        """
        return [
            self._tools[name]
            for name,metadata in self._metadata.items()
            if metadata.is_allowed_for(role)
        ]

    def contains(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return (
            f"ToolRegistry("
            f"tools={list(self._tools.keys())})"
        )


