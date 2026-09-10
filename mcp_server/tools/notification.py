from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mcp.server import MCPServer


# ---------------------------------------------------------------------------
# Mock notification backend
# 实际项目中可以替换为企业微信 / Slack / Teams / Email 等消息系统。
# ---------------------------------------------------------------------------

SUPPORTED_CHANNELS = {
    "oncall",
    "engineering",
    "incident",
}

SENT_NOTIFICATIONS: list[dict[str, Any]] = []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_notification_tools(mcp: MCPServer) -> None:
    """注册企业通知相关 MCP Tools。"""

    @mcp.tool()
    def send_notification(
        channel: str,
        recipients: list[str],
        message: str,
    ) -> dict[str, Any]:
        """
        向企业指定通信频道发送通知。

        该工具具有明显的外部副作用。
        在 Agent Runtime 中应视为高风险 Tool，
        由 PermissionPolicy、RiskPolicy 和 HITL 控制。

        Args:
            channel: 通知渠道，例如 oncall、engineering、incident。
            recipients: 接收人或团队列表。
            message: 通知内容。

        Returns:
            通知发送结果。
        """
        channel = channel.strip().lower()
        message = message.strip()

        cleaned_recipients = [
            recipient.strip()
            for recipient in recipients
            if recipient and recipient.strip()
        ]

        if channel not in SUPPORTED_CHANNELS:
            return {
                "success": False,
                "error": (
                    f"unsupported channel: {channel}; "
                    f"supported channels: {sorted(SUPPORTED_CHANNELS)}"
                ),
            }

        if not cleaned_recipients:
            return {
                "success": False,
                "error": "recipients cannot be empty",
            }

        if not message:
            return {
                "success": False,
                "error": "message cannot be empty",
            }

        notification = {
            "notification_id": f"MSG-{len(SENT_NOTIFICATIONS) + 1:04d}",
            "channel": channel,
            "recipients": cleaned_recipients,
            "message": message,
            "sent_at": _now_iso(),
            "status": "sent",
        }

        SENT_NOTIFICATIONS.append(notification)

        return {
            "success": True,
            "message": "notification sent successfully",
            "data": notification,
        }