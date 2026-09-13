from __future__ import annotations

from typing import Literal

from workflow.state import EnterpriseAgentState


SpecialistRoute = Literal[
    "recovery",
    "supervisor",
    "completed",
]


def specialist_router(
    state: EnterpriseAgentState,
) -> SpecialistRoute:
    """
    Specialist Outcome Router。

    只负责判断 Specialist 执行结果属于哪一种状态：

        failed
            ↓
        recovery

        completed + workflow_complete=False
            ↓
        supervisor

        completed + workflow_complete=True
            ↓
        completed

    Router 不负责：
        - 选择具体 Specialist
        - Tool Calling
        - Recovery Policy
        - Memory Persistence
        - Final Answer

    这些职责分别由：
        Supervisor / Recovery / Memory / Finalizer
    完成。
    """

    task_status = state.get(
        "task_status",
        "running",
    )

    # --------------------------------------------------------------
    # Specialist 执行失败
    # --------------------------------------------------------------

    if task_status == "failed":
        return "recovery"

    # --------------------------------------------------------------
    # Specialist 执行成功
    # --------------------------------------------------------------

    if task_status == "completed":

        if state.get(
            "workflow_complete",
            False,
        ):
            return "completed"

        return "supervisor"

    # --------------------------------------------------------------
    # Defensive fallback
    # --------------------------------------------------------------

    return "supervisor"