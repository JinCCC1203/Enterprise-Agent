from __future__ import annotations

from collections.abc import Iterable

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
    5. 创建受限的 ToolRegistryView
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._metadata: dict[str, ToolMetadata] = {}

    def register(
        self,
        tool: BaseTool,
        metadata: ToolMetadata,
    ) -> None:
        """
        注册一个 Tool。

        Tool 的真实名字以 tool.name 为准，
        metadata.name 必须与其保持一致。
        """

        if tool.name != metadata.name:
            raise ValueError(
                f"Tool name mismatch: "
                f"tool.name={tool.name!r}, "
                f"metadata.name={metadata.name!r}"
            )

        if tool.name in self._tools:
            raise ValueError(
                f"Tool already registered: {tool.name}"
            )

        self._tools[tool.name] = tool
        self._metadata[tool.name] = metadata

    def register_many(
        self,
        tools: Iterable[
            tuple[BaseTool, ToolMetadata]
        ],
    ) -> None:
        """
        批量注册 Tool。
        """

        for tool, metadata in tools:
            self.register(
                tool,
                metadata,
            )

    def unregister(
        self,
        tool_name: str,
    ) -> None:
        """
        删除 Tool。
        """

        self._tools.pop(
            tool_name,
            None,
        )

        self._metadata.pop(
            tool_name,
            None,
        )

    def get(
        self,
        tool_name: str,
    ) -> BaseTool:
        """
        根据 Tool name 获取 Tool。
        """

        try:
            return self._tools[tool_name]

        except KeyError as exc:
            raise KeyError(
                f"Tool not found: {tool_name}"
            ) from exc

    def get_metadata(
        self,
        tool_name: str,
    ) -> ToolMetadata:
        """
        根据 Tool name 获取 Tool Metadata。
        """

        try:
            return self._metadata[tool_name]

        except KeyError as exc:
            raise KeyError(
                f"Metadata not found: {tool_name}"
            ) from exc

    def get_all_tools(
        self,
    ) -> list[BaseTool]:
        """
        返回所有已注册 Tool。
        """

        return list(
            self._tools.values()
        )

    def get_all_metadata(
        self,
    ) -> list[ToolMetadata]:
        """
        返回所有 Tool Metadata。
        """

        return list(
            self._metadata.values()
        )

    def get_by_category(
        self,
        category: str,
    ) -> list[BaseTool]:
        """
        按 category 获取 Tool。
        """

        return [
            self._tools[name]
            for name, metadata in self._metadata.items()
            if metadata.category == category
        ]

    def get_by_source(
        self,
        source: str,
    ) -> list[BaseTool]:
        """
        按来源获取 Tool。
        """

        return [
            self._tools[name]
            for name, metadata in self._metadata.items()
            if metadata.source.value == source
        ]

    def get_by_permission(
        self,
        role: str,
    ) -> list[BaseTool]:
        """
        获取某个角色允许使用的 Tool。
        """

        return [
            self._tools[name]
            for name, metadata in self._metadata.items()
            if metadata.is_allowed_for(role)
        ]

    def create_view(
        self,
        tool_names: Iterable[str],
    ) -> ToolRegistryView:
        """
        创建一个受限的 ToolRegistryView。

        View 只允许访问指定 Tool，
        不会复制 Tool 实例，
        也不会修改 Global ToolRegistry。

        示例：

            knowledge_view = registry.create_view(
                {"rag_search"}
            )

            ticket_view = registry.create_view(
                {
                    "get_ticket",
                    "create_ticket",
                    "update_ticket",
                }
            )
        """

        return ToolRegistryView(
            registry=self,
            tool_names=tool_names,
        )

    def contains(
        self,
        tool_name: str,
    ) -> bool:
        return tool_name in self._tools

    def __len__(
        self,
    ) -> int:
        return len(self._tools)

    def __repr__(
        self,
    ) -> str:
        return (
            f"ToolRegistry("
            f"tools={list(self._tools.keys())})"
        )


class ToolRegistryView:
    """
    ToolRegistry 的只读受限视图。

    用于 Specialist Agent Tool Scope。

    Global Registry：

        ToolRegistry
        ├── rag_search
        ├── get_service_health
        ├── get_ticket
        ├── create_ticket
        ├── update_ticket
        ├── send_notification
        └── web_search

    Specialist View：

        Knowledge View
            └── rag_search

        Operations View
            └── get_service_health

        Ticket View
            ├── get_ticket
            ├── create_ticket
            └── update_ticket

        Research View
            └── web_search

    View 不拥有 Tool，只引用 Global Registry 中
    已存在的 Tool。

    因此不会复制：
        - Tool 实例
        - MCP Session
        - Tool Metadata
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        tool_names: Iterable[str],
    ) -> None:
        self._registry = registry

        # 使用 frozenset 保证：
        # 1. Scope 创建后不可修改
        # 2. Tool Name 唯一
        # 3. 可以安全地作为只读能力集合使用
        self._tool_names = frozenset(
            tool_names
        )

        # ----------------------------------------------------------
        # 创建 View 时立即验证 Tool 是否存在
        # ----------------------------------------------------------

        unknown_tools = {
            tool_name
            for tool_name in self._tool_names
            if not registry.contains(tool_name)
        }

        if unknown_tools:
            raise ValueError(
                "Unknown tools in registry scope: "
                f"{sorted(unknown_tools)}"
            )

    @property
    def tool_names(
        self,
    ) -> frozenset[str]:
        """
        返回当前 View 的 Tool Scope。
        """

        return self._tool_names

    def get_all_tools(
        self,
    ) -> list[BaseTool]:
        """
        返回当前 Scope 中的全部 Tool。
        """

        return [
            self._registry.get(tool_name)
            for tool_name in self._tool_names
        ]

    def get_tool(
        self,
        tool_name: str,
    ) -> BaseTool:
        """
        获取当前 Scope 内的 Tool。

        Scope 外的 Tool 不允许访问。
        """

        if tool_name not in self._tool_names:
            raise PermissionError(
                f"Tool '{tool_name}' is outside "
                "the current registry scope."
            )

        return self._registry.get(
            tool_name
        )

    def get_metadata(
        self,
        tool_name: str,
    ) -> ToolMetadata:
        """
        获取当前 Scope 内 Tool 的 Metadata。

        Scope 外的 Tool 不允许访问。
        """

        if tool_name not in self._tool_names:
            raise PermissionError(
                f"Tool '{tool_name}' is outside "
                "the current registry scope."
            )

        return self._registry.get_metadata(
            tool_name
        )

    def contains(
        self,
        tool_name: str,
    ) -> bool:
        """
        判断 Tool 是否属于当前 Scope。
        """

        return tool_name in self._tool_names

    def __len__(
        self,
    ) -> int:
        return len(
            self._tool_names
        )

    def __repr__(
        self,
    ) -> str:
        return (
            "ToolRegistryView("
            f"tools={sorted(self._tool_names)})"
        )