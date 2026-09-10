from __future__ import annotations

from typing import Any

from mcp.server import MCPServer


# ---------------------------------------------------------------------------
# Mock enterprise service state
# 实际项目中可以替换为 Prometheus / Kubernetes / Monitoring API 等真实后端。
# ---------------------------------------------------------------------------

SERVICE_STATUS: dict[str, dict[str, Any]] = {
    "order-service": {
        "service": "order-service",
        "status": "healthy",
        "latency_ms": 82,
        "error_rate": 0.002,
        "version": "v2.4.1",
        "owner": "platform-team",
    },
    "payment-service": {
        "service": "payment-service",
        "status": "degraded",
        "latency_ms": 641,
        "error_rate": 0.048,
        "version": "v3.1.0",
        "owner": "payment-team",
    },
    "user-service": {
        "service": "user-service",
        "status": "healthy",
        "latency_ms": 95,
        "error_rate": 0.001,
        "version": "v1.8.2",
        "owner": "identity-team",
    },
}


def register_service_tools(mcp: MCPServer) -> None:
    """注册企业服务监控相关 MCP Tools。"""

    @mcp.tool()
    def get_service_health(service_name: str) -> dict[str, Any]:
        """
        查询企业服务当前运行状态。

        该工具用于获取实时服务健康信息，例如服务状态、
        延迟、错误率、版本和责任团队。
        """
        service_name = service_name.strip()

        if not service_name:
            return {
                "success": False,
                "error": "service_name cannot be empty",
            }

        service = SERVICE_STATUS.get(service_name)

        if service is None:
            return {
                "success": False,
                "error": f"service not found: {service_name}",
                "available_services": sorted(SERVICE_STATUS.keys()),
            }

        return {
            "success": True,
            "data": {
                **service,
            },
        }