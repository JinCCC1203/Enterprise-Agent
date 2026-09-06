from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import ToolCallRequest

from tools_manager.metadata import ToolRiskLevel
from tools_manager.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class RiskResult:
    """
    risk_level:
        当前 Tool 的风险等级。
    requires_approval:
        是否需要额外人工确认。
    reason:
        风险判断原因，方便日志和审计。
    """

    risk_level: ToolRiskLevel
    requires_approval: bool
    reasons: tuple[str, ...]


class RiskPolicy:
    """
    对一次具体 Tool Call 进行风险评估。

    风险来源：
    1. ToolMetadata.risk_level
    2. Tool 名称 / 参数
    3. Runtime Context
    """

    def __init__(
            self,
            approval_levels: frozenset[ToolRiskLevel] | None = None,
    ) -> None:
        """
        approval_levels：
            哪些风险等级需要人工审批

        当前默认策略：
        LOW    -> 不需要审批
        MEDIUM -> 不需要审批
        HIGH   -> 需要审批
        """

        if approval_levels is None:
            approval_levels = frozenset(
                {
                    ToolRiskLevel.HIGH,
                }
            )

        self.approval_levels = approval_levels

    def evaluate(
        self,
        *,
        request: ToolCallRequest,
        registry: ToolRegistry,
    ) -> RiskResult:

        tool_call = request.tool_call

        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})

        # 获取 Metadata
        try:
            metadata = registry.get_metadata(tool_name)

        except KeyError:
            return RiskResult(
                risk_level=ToolRiskLevel.HIGH,
                requires_approval=True,
                reasons=(
                    f"metadata not found for tool '{tool_name}'",
                    "execution requires human approval",
                ),
            )

        risk_level = metadata.risk_level

        # 收集所有审批原因
        reasons: list[str] = []

        # Tool 静态风险等级
        if risk_level in self.approval_levels:
            reasons.append(
                f"tool risk level is '{risk_level.value}'"
            )

        # Tool 名称 / 参数
        if self._is_sensitive_operation(
            tool_name=tool_name,
            tool_args=tool_args,
        ):
            reasons.append(
                "tool call is a sensitive operation"
            )

        # Runtime Context
        user_role = self._get_user_role(request)

        if self._requires_approval_for_role(
            user_role=user_role,
            tool_name=tool_name,
        ):
            reasons.append(
                f"role '{user_role}' requires approval "
                f"for tool '{tool_name}'"
            )

        # 统一计算最终结果
        requires_approval = bool(reasons)

        # 如果没有触发任何规则，也保留基础信息
        if not reasons:
            reasons.append(
                f"tool risk level is '{risk_level.value}' "
                f"and no additional approval rule was triggered"
            )

        return RiskResult(
            risk_level=risk_level,
            requires_approval=requires_approval,
            reasons=tuple(reasons),
        )

    @staticmethod
    def _is_sensitive_operation(
        *,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> bool:

        sensitive_keywords = (
            "delete",
            "remove",
            "destroy",
            "update",
            "send",
        )

        if any(
            keyword in tool_name.lower()
            for keyword in sensitive_keywords
        ):
            return True

        if tool_args.get("batch", False):
            return True

        return False

    @staticmethod
    def _get_user_role(
        request: ToolCallRequest,
    ) -> str:

        context = request.runtime.context

        return getattr(
            context,
            "user_role",
            "unknown",
        )

    @staticmethod
    def _requires_approval_for_role(
        *,
        user_role: str,
        tool_name: str,
    ) -> bool:

        if user_role == "employee":
            sensitive_tools = {
                "send_email",
                "delete_repository",
            }

            return tool_name in sensitive_tools


        return False