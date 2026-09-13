from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from workflow.state import EnterpriseAgentState


RecoveryAction = Literal[
    "retry_agent",
    "reroute",
    "human_review",
    "failed",
]


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """
    Workflow-level Recovery Decision。

    Recovery 不负责：

        Tool Retry
        Model Retry
        Permission Check
        Risk Check
        HITL Middleware

    Recovery 只负责：

        Specialist 最终失败之后，
        Workflow 下一步如何恢复。
    """

    action: RecoveryAction
    reason: str


class RecoveryPolicy:
    """
    确定性的 Workflow Recovery Policy。

    核心规则：

        高风险 / 权限 / 安全
            → human_review

        Tool / MCP / Infrastructure
            → reroute → Supervisor

        Specialist 自身失败
            → retry_agent

        无法安全判断
            → human_review

        Recovery 次数耗尽
            → failed
    """

    DEFAULT_MAX_RECOVERY_ATTEMPTS = 2

    # ==============================================================
    # High-risk / Security
    # ==============================================================

    HIGH_RISK_KEYWORDS = frozenset(
        {
            "permission denied",
            "forbidden",
            "unauthorized",
            "authorization",
            "policy violation",
            "security violation",
            "security error",
            "high risk",
            "approval required",
        }
    )

    # ==============================================================
    # Tool / MCP / Infrastructure
    # ==============================================================

    REROUTE_KEYWORDS = frozenset(
        {
            "timeout",
            "timed out",
            "connection refused",
            "connection reset",
            "connection error",
            "network error",
            "service unavailable",
            "tool unavailable",
            "mcp error",
            "mcp unavailable",
            "upstream unavailable",
            "gateway timeout",
            "server unavailable",
        }
    )

    def __init__(
        self,
        *,
        max_recovery_attempts: int = DEFAULT_MAX_RECOVERY_ATTEMPTS,
    ) -> None:
        if max_recovery_attempts < 0:
            raise ValueError(
                "max_recovery_attempts must be >= 0."
            )

        self.max_recovery_attempts = max_recovery_attempts

    # ==============================================================
    # Recovery Decision
    # ==============================================================

    def decide(
        self,
        state: EnterpriseAgentState,
    ) -> RecoveryDecision:
        """
        根据当前 Workflow Failure
        决定 Recovery Action。
        """

        error = (
            state.get("error") or ""
        ).strip()

        failed_node = state.get(
            "last_failed_node"
        )

        failed_tool = state.get(
            "last_failed_tool"
        )

        current_agent = state.get(
            "current_agent"
        )

        recovery_attempts = state.get(
            "recovery_attempts",
            0,
        )

        # ==========================================================
        # 0. Recovery 输入异常
        # ==========================================================

        if (
            not error
            and not failed_node
        ):
            return RecoveryDecision(
                action="failed",
                reason=(
                    "Recovery was entered without "
                    "a structured workflow failure."
                ),
            )

        # ==========================================================
        # 1. Recovery 次数已耗尽
        #
        # recovery_attempts 表示已经执行过的
        # Workflow Recovery Action 数量。
        #
        # max=2：
        #
        # attempts=0 → 第一次 Recovery
        # attempts=1 → 第二次 Recovery
        # attempts=2 → Terminal Failure
        # ==========================================================

        if (
            recovery_attempts
            >= self.max_recovery_attempts
        ):
            return RecoveryDecision(
                action="failed",
                reason=(
                    "Maximum workflow recovery "
                    "attempts have been exceeded."
                ),
            )

        normalized_error = error.lower()

        # ==========================================================
        # 2. High-risk / Permission / Security
        # ==========================================================

        if self._contains_keyword(
            normalized_error,
            self.HIGH_RISK_KEYWORDS,
        ):
            return RecoveryDecision(
                action="human_review",
                reason=(
                    "The failure involves authorization, "
                    "security, policy, or another high-risk "
                    "condition that requires human review."
                ),
            )

        # ==========================================================
        # 3. Tool / MCP / Infrastructure
        #
        # last_failed_tool 非 None 时，
        # 优先认为这是 Tool-level failure。
        # ==========================================================

        if (
            failed_tool is not None
            or self._contains_keyword(
                normalized_error,
                self.REROUTE_KEYWORDS,
            )
        ):
            return RecoveryDecision(
                action="reroute",
                reason=(
                    "The failure appears to originate "
                    "from a Tool, MCP server, network, "
                    "or external service. Returning control "
                    "to Supervisor for re-planning."
                ),
            )

        # ==========================================================
        # 4. Specialist 自身失败
        # ==========================================================

        if current_agent is not None:
            return RecoveryDecision(
                action="retry_agent",
                reason=(
                    "The current Specialist Agent failed "
                    "without clear evidence of an external "
                    "Tool or infrastructure failure. "
                    "Retrying the original Specialist Agent."
                ),
            )

        # ==========================================================
        # 5. Unknown Failure
        # ==========================================================

        return RecoveryDecision(
            action="human_review",
            reason=(
                "The failure cannot be safely classified "
                "for automatic recovery."
            ),
        )

    # ==============================================================
    # Keyword Helper
    # ==============================================================

    @staticmethod
    def _contains_keyword(
        text: str,
        keywords: frozenset[str],
    ) -> bool:
        """
        判断错误文本是否包含指定关键词。
        """

        return any(
            keyword in text
            for keyword in keywords
        )


# ==============================================================
# Recovery Node Factory
# ==============================================================

def create_recovery_node(
    *,
    recovery_policy: RecoveryPolicy | None = None,
):
    """
    创建 LangGraph Recovery Node。

    Recovery Node：

        EnterpriseAgentState
              ↓
        RecoveryPolicy
              ↓
        RecoveryDecision
              ↓
        更新 Workflow State
    """

    policy = (
        recovery_policy
        or RecoveryPolicy()
    )

    async def recovery_node(
        state: EnterpriseAgentState,
    ) -> dict[str, Any]:

        current_attempts = state.get(
            "recovery_attempts",
            0,
        )

        current_agent = state.get(
            "current_agent"
        )

        decision = policy.decide(
            state
        )

        # ==========================================================
        # 1. Terminal Failure
        #
        # Recovery 已耗尽：
        #
        #     task_status = failed
        #     workflow_complete = False
        #     current_agent = None
        #     next_agent = end
        #
        # 不允许 Supervisor 再次规划。
        # ==========================================================

        if decision.action == "failed":

            return {
                "recovery_status": "failed",
                "recovery_reason": decision.reason,
                "recovery_attempts": current_attempts,
                "resume_required": False,
                "task_status": "failed",
                "workflow_complete": False,
                "current_agent": None,
                "next_agent": "end",
                "handoff_task": None,
            }

        # ==========================================================
        # 2. 本次执行 Recovery Action
        #
        # 只有真正执行了一次 Recovery Action，
        # 才增加 recovery_attempts。
        # ==========================================================

        next_attempts = (
            current_attempts + 1
        )

        # ==========================================================
        # 3. Retry Original Specialist
        # ==========================================================

        if decision.action == "retry_agent":

            return {
                "recovery_status": "retry",
                "recovery_reason": decision.reason,
                "recovery_attempts": next_attempts,
                "resume_required": False,
                "task_status": "running",
                "workflow_complete": False,
                "next_agent": current_agent,
            }

        # ==========================================================
        # 4. Reroute → Supervisor
        # ==========================================================

        if decision.action == "reroute":

            return {
                "recovery_status": "reroute",
                "recovery_reason": decision.reason,
                "recovery_attempts": next_attempts,
                "resume_required": False,
                "task_status": "running",
                "workflow_complete": False,
                "next_agent": "supervisor",
                "handoff_task": None,
            }

        # ==========================================================
        # 5. Human Review
        # ==========================================================

        if decision.action == "human_review":

            return {
                "recovery_status": "human_review",
                "recovery_reason": decision.reason,
                "recovery_attempts": next_attempts,
                "resume_required": True,
                "task_status": "interrupted",
                "workflow_complete": False,
                "next_agent": "human_review",
            }

        # ==========================================================
        # 6. Defensive Fallback
        # ==========================================================

        return {
            "recovery_status": "failed",
            "recovery_reason": (
                "Unknown recovery action."
            ),
            "recovery_attempts": current_attempts,
            "resume_required": False,
            "task_status": "failed",
            "workflow_complete": False,
            "current_agent": None,
            "next_agent": "end",
            "handoff_task": None,
        }

    return recovery_node