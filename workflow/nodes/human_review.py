from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from workflow.state import EnterpriseAgentState


def create_human_review_node():
    """
    创建 Workflow-level Human Review Node。

    用途：

        Recovery
           ↓
        human_review
           ↓
        interrupt()
           ↓
        等待人工决策
           ↓
        Command(resume=...)
           ↓
        继续当前 Workflow

    这里的 Human Review 与
    HumanInTheLoopMiddleware 不同：

        HumanInTheLoopMiddleware
            → Tool Call approval

        Human Review Node
            → Workflow Recovery decision
    """

    async def human_review_node(
        state: EnterpriseAgentState,
    ) -> dict[str, Any]:

        failed_node = state.get(
            "last_failed_node"
        )

        failed_tool = state.get(
            "last_failed_tool"
        )

        error = state.get(
            "error"
        )

        recovery_reason = state.get(
            "recovery_reason"
        )

        # ==========================================================
        # interrupt payload
        #
        # 这里不执行任何外部副作用。
        # ==========================================================

        review_request = {
            "type": "workflow_recovery_review",
            "message": (
                "The Agent Workflow requires "
                "human intervention before recovery."
            ),
            "failed_node": failed_node,
            "failed_tool": failed_tool,
            "error": error,
            "recovery_reason": recovery_reason,
            "allowed_decisions": [
                "retry",
                "reroute",
                "reject",
            ],
        }

        decision = interrupt(
            review_request
        )

        # ==========================================================
        # Resume 后，decision 即为：
        #
        # {
        #     "action": "retry"
        # }
        #
        # 或：
        #
        # {
        #     "action": "reroute"
        # }
        #
        # 或：
        #
        # {
        #     "action": "reject"
        # }
        # ==========================================================

        if not isinstance(
            decision,
            dict,
        ):
            raise ValueError(
                "Human review resume value must "
                "be a dictionary."
            )

        action = decision.get(
            "action"
        )

        if action not in {
            "retry",
            "reroute",
            "reject",
        }:
            raise ValueError(
                "Invalid human recovery decision: "
                f"{action!r}"
            )

        # ==========================================================
        # 将人工决策写入 State
        # ==========================================================

        if action == "retry":

            return {
                "approval_status": "approved",
                "recovery_status": "human_retry",
                "resume_required": False,
                "task_status": "running",
            }

        if action == "reroute":

            return {
                "approval_status": "approved",
                "recovery_status": "human_reroute",
                "resume_required": False,
                "task_status": "running",
            }

        # reject
        return {
            "approval_status": "rejected",
            "recovery_status": "human_rejected",
            "resume_required": False,
            "task_status": "failed",
        }

    return human_review_node