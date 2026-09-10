from __future__ import annotations

from dataclasses import dataclass

from tools_manager.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class PermissionResult:
    """
        权限检查结果。

        allowed:
            是否允许使用该工具

        reason:
            拒绝时给出原因，方便日志、调试和后续审计
    """

    allowed: bool
    reason: str = ""

class PermissionPolicy:
    """
    职责：
    1. 从运行时 ToolRegistry 获取 ToolMetadata
    2. 根据 user_role 判断是否允许使用 Tool
    3. 返回 PermissionResult
    """

    def check(
        self,
        *,
        user_role: str,
        tool_name: str,
        registry: ToolRegistry,
    ) -> PermissionResult:

        if not user_role:
            return PermissionResult(
                allowed=False,
                reason="user role is empty",
            )

        # 从运行时 Registry 获取 Metadata
        try:
            metadata = registry.get_metadata(tool_name)
        except KeyError:
            return PermissionResult(
                allowed=False,
                reason=(
                    f"metadata not found "
                    f"for tool '{tool_name}'"
                ),
            )

        # 权限判断
        # Metadata 是纯数据对象，
        # 具体权限语义由 metadata.is_allowed_for() 提供。
        allowed = metadata.is_allowed_for(user_role)

        if allowed:
            return PermissionResult(
                allowed=True,
                reason=(
                    f"user role '{user_role}' "
                    f"is allowed to use "
                    f"tool '{tool_name}'"
                ),
            )

        return PermissionResult(
            allowed=False,
            reason=(
                f"user role '{user_role}' "
                f"is not allowed to use "
                f"tool '{tool_name}'"
            ),
        )

    def is_allowed(
        self,
        *,
        user_role: str,
        tool_name: str,
        registry: ToolRegistry,
    ) -> bool:

        return self.check(
            user_role=user_role,
            tool_name=tool_name,
            registry=registry,
        ).allowed


