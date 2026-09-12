from __future__ import annotations

from langchain_core.tools import BaseTool

from policies.permission import PermissionPolicy
from tools_manager.registry import ToolRegistryView


class PermissionBasedToolExposure:
    """
    基于：
        Specialist Tool Scope
        +
        Runtime Role Permission
    决定哪些 Tool 可以暴露给 LLM。

    不负责最终 Tool Call，最终 Tool Call 仍然由 LLM 决定。
    """

    def __init__(
        self,
        *,
        registry_view: ToolRegistryView,
        permission_policy: PermissionPolicy,
    ) -> None:

        self.registry_view = registry_view
        self.permission_policy = permission_policy

    def expose(
        self,
        *,
        role: str,
    ) -> list[BaseTool]:
        """
        在当前 Specialist Scope 内，
        根据用户角色决定可暴露 Tool。
        """

        exposed_tools: list[BaseTool] = []

        for tool in self.registry_view.get_all_tools():

            permission_result = (
                self.permission_policy.check(
                    user_role=role,
                    tool_name=tool.name,
                    registry=self.registry_view,
                )
            )

            if not permission_result.allowed:
                continue

            exposed_tools.append(tool)

        return exposed_tools

