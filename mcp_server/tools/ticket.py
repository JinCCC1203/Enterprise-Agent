from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mcp.server import MCPServer


# ---------------------------------------------------------------------------
# Mock enterprise ticket system
# 实际项目中可以替换为 Jira / ServiceNow / 企业内部工单系统 API。
# ---------------------------------------------------------------------------

TICKETS: dict[str, dict[str, Any]] = {
    "INC-2026-1001": {
        "ticket_id": "INC-2026-1001",
        "title": "支付服务延迟升高",
        "description": "payment-service latency above threshold",
        "priority": "P1",
        "status": "investigating",
        "service_name": "payment-service",
        "owner": "payment-team",
        "comments": [
            {
                "timestamp": "2026-09-10T08:30:00+00:00",
                "comment": "开始排查支付服务延迟问题。",
            }
        ],
        "created_at": "2026-09-10T08:20:00+00:00",
        "updated_at": "2026-09-10T08:30:00+00:00",
    }
}


VALID_PRIORITIES = {"P1", "P2", "P3", "P4"}

VALID_STATUSES = {
    "open",
    "investigating",
    "resolved",
    "closed",
}


def _now_iso() -> str:
    """返回当前 UTC ISO 时间。"""
    return datetime.now(timezone.utc).isoformat()


def register_ticket_tools(mcp: MCPServer) -> None:
    """注册企业工单相关 MCP Tools。"""

    @mcp.tool()
    def get_ticket(ticket_id: str) -> dict[str, Any]:
        """
        查询企业工单或 Incident 的当前状态。

        Args:
            ticket_id: 工单编号，例如 INC-2026-1001。

        Returns:
            工单详细信息。
        """
        ticket_id = ticket_id.strip()

        if not ticket_id:
            return {
                "success": False,
                "error": "ticket_id cannot be empty",
            }

        ticket = TICKETS.get(ticket_id)

        if ticket is None:
            return {
                "success": False,
                "error": f"ticket not found: {ticket_id}",
            }

        return {
            "success": True,
            "data": {
                **ticket,
            },
        }

    @mcp.tool()
    def create_ticket(
        title: str,
        description: str,
        priority: str,
        service_name: str,
    ) -> dict[str, Any]:
        """
        创建新的企业 Incident / 工单。

        这是一个具有外部副作用的写操作。
        在 Agent Runtime 中应由 PermissionPolicy、
        RiskPolicy 和 Human-in-the-Loop 进行治理。

        Args:
            title: 工单标题。
            description: 问题详细描述。
            priority: 优先级，可选 P1/P2/P3/P4。
            service_name: 受影响的服务名称。
        Returns:
            创建结果以及新工单信息。
        """
        title = title.strip()
        description = description.strip()
        priority = priority.strip().upper()
        service_name = service_name.strip()

        if not title:
            return {
                "success": False,
                "error": "title cannot be empty",
            }

        if not description:
            return {
                "success": False,
                "error": "description cannot be empty",
            }

        if priority not in VALID_PRIORITIES:
            return {
                "success": False,
                "error": (
                    f"invalid priority: {priority}; "
                    f"must be one of {sorted(VALID_PRIORITIES)}"
                ),
            }

        if not service_name:
            return {
                "success": False,
                "error": "service_name cannot be empty",
            }

        # 简单生成递增的 mock ticket ID。
        sequence = len(TICKETS) + 1001
        ticket_id = f"INC-2026-{sequence}"

        now = _now_iso()

        ticket = {
            "ticket_id": ticket_id,
            "title": title,
            "description": description,
            "priority": priority,
            "status": "open",
            "service_name": service_name,
            "owner": None,
            "comments": [],
            "created_at": now,
            "updated_at": now,
        }

        TICKETS[ticket_id] = ticket

        return {
            "success": True,
            "message": "ticket created successfully",
            "data": {
                **ticket,
            },
        }

    @mcp.tool()
    def update_ticket(
        ticket_id: str,
        status: str,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """
        更新已有工单的状态和处理备注。

        该操作设计为幂等更新：
        对相同 ticket_id 设置相同状态不会创建重复工单。

        在 Agent Runtime 中可以用于演示：
        Permission Check + Retry + Logging + Idempotency。

        Args:
            ticket_id: 工单编号。
            status: 新状态，可选 open/investigating/resolved/closed。
            comment: 可选的处理说明。
        Returns:
            更新后的工单信息。
        """
        ticket_id = ticket_id.strip()
        status = status.strip().lower()

        if not ticket_id:
            return {
                "success": False,
                "error": "ticket_id cannot be empty",
            }

        if status not in VALID_STATUSES:
            return {
                "success": False,
                "error": (
                    f"invalid status: {status}; "
                    f"must be one of {sorted(VALID_STATUSES)}"
                ),
            }

        ticket = TICKETS.get(ticket_id)

        if ticket is None:
            return {
                "success": False,
                "error": f"ticket not found: {ticket_id}",
            }

        now = _now_iso()

        # 幂等更新：即使状态没有变化，也允许安全返回。
        ticket["status"] = status
        ticket["updated_at"] = now

        if comment is not None:
            comment = comment.strip()

            if comment:
                ticket.setdefault("comments", []).append(
                    {
                        "timestamp": now,
                        "comment": comment,
                    }
                )

        return {
            "success": True,
            "message": "ticket updated successfully",
            "data": {
                **ticket,
            },
        }