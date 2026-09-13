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

        - Tool Retry
        - Model Retry
        - Permission Check
        - Risk Check
        - HITL Middleware

    这些职责分别属于：

        LangChain Middleware
        Tool Governance
        HumanInTheLoopMiddleware

    Recovery 只负责：

        当前 Specialist 已经最终失败以后，
        Workflow 下一步应该如何恢复。

    action:

        retry_agent
            → 回到原 Specialist

        reroute
            → 返回 Supervisor 重新规划

        human_review
            → 进入 Human Review / Interrupt

        failed
            → 本次 Workflow 无法自动恢复，
              进入最终持久化阶段
    """

    action: RecoveryAction
    reason: str


class RecoveryPolicy:
    """
    确定性的 Workflow Recovery Policy。

    核心原则：

        1. 高风险 / 权限 / 安全问题
           → human_review

        2. Tool / MCP / 基础设施问题
           → reroute → Supervisor

        3. Specialist 自身的未知 transient failure
           → retry_agent

        4. 无法安全判断
           → human_review

        5. Workflow Recovery 次数耗尽
           → failed

    注意：

        本 Policy 不读取：

            tool_retry_count
            model_retry_count

        因为 Tool / Model Retry 属于 Middleware
        内部执行层。

        本 Policy 只读取 Workflow State。
    """

    DEFAULT_MAX_RECOVERY_ATTEMPTS = 2

    # --------------------------------------------------------------
    # 高风险 / 安全 / 权限错误
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Tool / MCP / Infrastructure 类错误
    #
    # 这些问题通常说明：
    #
    #   当前 Specialist 依赖的能力不可用
    #
    # 因此不应该不断执行同一个 Specialist，
    # 而应该返回 Supervisor 重新规划。
    # --------------------------------------------------------------

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
        max_recovery_attempts: int = (
            DEFAULT_MAX_RECOVERY_ATTEMPTS
        ),
    ) -> None:

        if max_recovery_attempts < 0:
            raise ValueError(
                "max_recovery_attempts must be >= 0."
            )

        self.max_recovery_attempts = (
            max_recovery_attempts
        )

    def decide(
        self,
        state: EnterpriseAgentState,
    ) -> RecoveryDecision:
        """
        根据当前 Workflow failure
        决定下一步 Recovery Action。
        """

        error = (
            state.get("error") or ""
        ).strip()

        failed_node = (
            state.get("last_failed_node")
        )

        failed_tool = (
            state.get("last_failed_tool")
        )

        current_agent = (
            state.get("current_agent")
        )

        recovery_attempts = state.get(
            "recovery_attempts",
            0,
        )

        # ==========================================================
        # 0. Recovery 输入本身不完整
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
        # 1. Recovery 次数达到上限
        #
        # 注意：
        #
        # recovery_attempts 是 Workflow-level counter。
        #
        # 它不是：
        #   Tool retry count
        #   Model retry count
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
        # 2. 高风险 / 权限 / 安全问题
        #
        # 不能通过 reroute 绕过安全边界。
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
        # 3. Tool / MCP / Infrastructure Failure
        #
        # Tool 已经经过 Middleware Retry，
        # 仍然失败 → 当前 Specialist 依赖的能力不可用。
        #
        # 不继续死循环执行当前 Specialist，
        # 返回 Supervisor 重新规划。
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
        # 4. Specialist Node 自身失败
        #
        # 如果没有明显的 Tool / Infrastructure failure，
        # 可以做一次 Workflow-level Agent retry。
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
        #
        # 无法安全判断时，保守交给人工。
        # ==========================================================

        return RecoveryDecision(
            action="human_review",
            reason=(
                "The failure cannot be safely classified "
                "for automatic recovery."
            ),
        )

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

    注意：

        Recovery Node 不直接决定具体 Specialist。

        retry_agent：
            由 recovery_router 根据
            last_failed_node 决定回哪个 Specialist。

        reroute：
            统一返回 Supervisor。

        human_review：
            返回 Human Review Node。

        failed：
            返回 Memory Persist。
    """

    policy = (
        recovery_policy
        or RecoveryPolicy()
    )

    async def recovery_node(
        state: EnterpriseAgentState,
    ) -> dict[str, Any]:

        decision = policy.decide(
            state
        )

        current_attempts = state.get(
            "recovery_attempts",
            0,
        )

        next_attempts = (
            current_attempts + 1
        )

        # ==========================================================
        # 1. Retry original Specialist
        # ==========================================================

        if decision.action == "retry_agent":

            return {
                "recovery_status": "retry",
                "recovery_reason": decision.reason,
                "recovery_attempts": next_attempts,
                "resume_required": False,
                "task_status": "running",
            }

        # ==========================================================
        # 2. Reroute → Supervisor
        # ==========================================================

        if decision.action == "reroute":

            return {
                "recovery_status": "reroute",
                "recovery_reason": decision.reason,
                "next_agent": "supervisor",
                "recovery_attempts": next_attempts,
                "resume_required": False,
                "task_status": "running",
            }

        # ==========================================================
        # 3. Human Review
        #
        # 不在这里直接 interrupt。
        #
        # Recovery Router 会把它路由到：
        #
        #     human_review
        #
        # 由独立 Human Review Node 负责 interrupt/resume。
        # ==========================================================

        if decision.action == "human_review":

            return {
                "recovery_status": "human_review",
                "recovery_reason": decision.reason,
                "recovery_attempts": next_attempts,
                "resume_required": True,
                "task_status": "interrupted",
            }

        # ==========================================================
        # 4. Terminal Failure
        #
        # 不在 Recovery Node 直接 END。
        #
        # Recovery Router 会：
        #
        #     failed
        #       ↓
        #     memory_persist
        #       ↓
        #     END
        #
        # 这样失败 Workflow 中仍然可以提取有效长期记忆。
        # ==========================================================

        return {
            "recovery_status": "failed",
            "recovery_reason": decision.reason,
            "recovery_attempts": next_attempts,
            "resume_required": False,
            "task_status": "failed",
        }

    return recovery_node