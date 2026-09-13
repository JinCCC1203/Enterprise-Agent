from __future__ import annotations

from typing import Literal

from workflow.state import EnterpriseAgentState


RecoveryRoute = Literal[
    "knowledge_agent",
    "operations_agent",
    "ticket_agent",
    "research_agent",
    "supervisor",
    "human_review",
    "memory_persist",
]


_FAILED_NODE_TO_AGENT: dict[
    str,
    RecoveryRoute,
] = {
    "knowledge_agent": "knowledge_agent",
    "operations_agent": "operations_agent",
    "ticket_agent": "ticket_agent",
    "research_agent": "research_agent",
}


def recovery_router(
    state: EnterpriseAgentState,
) -> RecoveryRoute:
    """
    Recovery Node → Graph Route。

    RecoveryPolicy：
        决定恢复策略。

    Recovery Router：
        将策略转换为 Graph Edge。

    retry:
        → 原 Specialist

    reroute:
        → Supervisor

    human_review:
        → Human Review Node

    failed:
        → Memory Persist
    """

    recovery_status = state.get(
        "recovery_status"
    )

    # --------------------------------------------------------------
    # retry
    # --------------------------------------------------------------

    if recovery_status == "retry":

        failed_node = state.get(
            "last_failed_node"
        )

        if not failed_node:
            raise ValueError(
                "Recovery retry requires "
                "last_failed_node."
            )

        route = _FAILED_NODE_TO_AGENT.get(
            failed_node
        )

        if route is None:
            raise ValueError(
                "Unsupported Specialist Agent: "
                f"{failed_node!r}"
            )

        return route

    # --------------------------------------------------------------
    # reroute
    # --------------------------------------------------------------

    if recovery_status == "reroute":
        return "supervisor"

    # --------------------------------------------------------------
    # human review
    # --------------------------------------------------------------

    if recovery_status == "human_review":
        return "human_review"

    # --------------------------------------------------------------
    # terminal failure
    # --------------------------------------------------------------

    if recovery_status == "failed":
        return "memory_persist"

    raise ValueError(
        "Invalid recovery_status: "
        f"{recovery_status!r}"
    )