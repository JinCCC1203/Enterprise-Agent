from __future__ import annotations

from typing import Literal

from workflow.state import EnterpriseAgentState


SupervisorRoute = Literal[
    "knowledge_agent",
    "operations_agent",
    "ticket_agent",
    "research_agent",
    "end",
]


_ALLOWED_ROUTES: frozenset[str] = frozenset(
    {
        "knowledge_agent",
        "operations_agent",
        "ticket_agent",
        "research_agent",
        "end",
    }
)


def supervisor_router(
    state: EnterpriseAgentState,
) -> SupervisorRoute:
    """
    Supervisor Conditional Router。

    Supervisor Node:
        User Query
            ↓
        LLM decision
            ↓
        state["next_agent"]

    Router:
        state["next_agent"]
            ↓
        Conditional Edge
            ↓
        Specialist Agent / END

    职责边界：

    Supervisor:
        负责理解任务并做出 Agent 路由决策。

    Router:
        只负责把 Supervisor 的决策转换成
        LangGraph 的 conditional route。

    Router 不：
        - 调用 LLM
        - 调用 Tool
        - 做语义分类
        - 修改 next_agent
        - 重新判断任务应该交给谁
    """

    next_agent = state.get("next_agent")

    if next_agent is None:
        raise ValueError(
            "Supervisor did not provide next_agent."
        )

    if next_agent not in _ALLOWED_ROUTES:
        raise ValueError(
            f"Unknown supervisor route: {next_agent!r}. "
            f"Allowed routes: {sorted(_ALLOWED_ROUTES)}"
        )

    return next_agent  # type: ignore[return-value]