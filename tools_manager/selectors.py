from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import BaseTool

from policies.permission import PermissionPolicy
from tools_manager.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class SelectionContext:
    """
    Tool Selector 使用的上下文。
    """
    role: str
    task_type: str | None = None
    query: str = ""
"""
query
 ↓
Intent / Task Classification
 ↓
category / tags / tool routing
 ↓
Selector
"""


class RuleBasedToolSelector:
    """
    基于规则的 Tool Selector。

    执行流程：

        Runtime Registry
              ↓
        Permission Filter
              ↓
        Metadata Rule Filter
              ↓
        Candidate Tools

    Selector 不负责最终 Tool Call。
    最终 Tool Call 仍然由 LLM 决定。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission_policy: PermissionPolicy,
    ) -> None:

        self.registry = registry
        self.permission_policy = permission_policy

    def select(
        self,
        context: SelectionContext,
    ) -> list[BaseTool]:

        selected_tools: list[BaseTool] = []

        # 从当前运行时 Registry 获取全部 Tool

        tools = self.registry.get_all_tools()

        # 逐个执行规则过滤
        for tool in tools:
            # Permission policy
            permission_result = self.permission_policy.check(
                user_role=context.role,
                tool_name=tool.name,
                registry=self.registry,
            )

            if not permission_result.allowed:
                continue

            # 获取 Metadata
            metadata = self.registry.get_metadata(
                tool.name
            )

            # Task Type
            if not self._match_task_type(
                task_type=context.task_type,
                metadata=metadata,
            ):
                continue

            # Category
            if not self._match_category(
                context=context,
                metadata=metadata,
            ):
                continue

            # Tags
            if not self._match_tags(
                context=context,
                metadata=metadata,
            ):
                continue

            # 通过所有规则

            selected_tools.append(tool)

        return selected_tools

    @staticmethod
    def _match_task_type(
        *,
        task_type: str | None,
        metadata,
    ) -> bool:
        """
        后续 metadata 增加：

            task_types=frozenset(...)

        后即可正式启用。
        """
        supported_task_types = getattr(
            metadata,
            "task_types",
            None,
        )

        if not supported_task_types:
            return True

        if task_type is None:
            return True

        return task_type in supported_task_types

    @staticmethod
    def _match_category(
        *,
        context: SelectionContext,
        metadata,
    ) -> bool:
        """
        暂时不根据 category 做强制过滤。

        例如：

            metadata.category == "knowledge"
            metadata.category == "development"

        后续可以根据 task_type / intent 做 category 路由。
        """
        return True

    @staticmethod
    def _match_tags(
        *,
        context: SelectionContext,
        metadata,
    ) -> bool:
        """
        暂时不根据 tags 做强制过滤。

        后续可以根据 query / intent 做 tag matching。
        """
        return True