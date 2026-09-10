from __future__ import annotations

from csv import excel
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from policies.permission import PermissionPolicy
from tools_manager.registry import ToolRegistry


class PermissionBasedToolExposure:
    """
    不负责最终 Tool Call，之决定哪些 tool 暴露给llm
    最终 Tool Call 仍然由 LLM 决定。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission_policy: PermissionPolicy,
    ) -> None:

        self.registry = registry
        self.permission_policy = permission_policy

    def expose(
            self,
            *,
            role: str,
    ) -> list[BaseTool]:

        exposed_tools: list[BaseTool] = []

        # 从当前运行时 Registry 获取全部 Tool

        tools = self.registry.get_all_tools()

        # 逐个执行规则过滤
        for tool in tools:
            # Permission policy
            permission_result = self.permission_policy.check(
                user_role=role,
                tool_name=tool.name,
                registry=self.registry,
            )

            if not permission_result.allowed:
                continue

            exposed_tools.append(tool)

        return exposed_tools

