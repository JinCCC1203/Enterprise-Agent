from __future__ import annotations

from typing import Literal

from workflow.state import EnterpriseAgentState


HumanReviewRoute = Literal[
    "knowledge_agent",
    "operations_agent",
    "ticket_agent",
    "research_agent",
    "supervisor",
    "memory_persist",
]


_FAILED_NODE_TO_AGENT: dict[
    str,
    HumanReviewRoute,
] = {
    "knowledge_agent": "knowledge_agent",
    "operations_agent": "operations_agent",
    "ticket_agent": "ticket_agent",
    "research_agent": "research_agent",
}


def human_review_router(
    state: EnterpriseAgentState,
) -> HumanReviewRoute:
    """
    Human Review → 下一步 Workflow Route。

    人工决定：

        retry
            → 原 Specialist

        reroute
            → Supervisor

        reject
            → Memory Persist
    """

    recovery_status = state.get(
        "recovery_status"
    )

    # ==============================================================
    # 1. Human approve retry
    # ==============================================================

    if recovery_status == "human_retry":

        failed_node = state.get(
            "last_failed_node"
        )

        if not failed_node:
            raise ValueError(
                "Human selected retry, but "
                "last_failed_node is missing."
            )

        route = _FAILED_NODE_TO_AGENT.get(
            failed_node
        )

        if route is None:
            raise ValueError(
                "Unsupported failed node for "
                f"human retry: {failed_node!r}"
            )

        return route

    # ==============================================================
    # 2. Human approve reroute
    # ==============================================================

    if recovery_status == "human_reroute":
        return "supervisor"

    # ==============================================================
    # 3. Human reject
    # ==============================================================

    if recovery_status == "human_rejected":
        return "memory_persist"

    raise ValueError(
        "Invalid human review recovery status: "
        f"{recovery_status!r}"
    )