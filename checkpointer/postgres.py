from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import (
    AsyncPostgresSaver,
)


@asynccontextmanager
async def create_checkpointer(
    database_url: str,
) -> AsyncIterator[AsyncPostgresSaver]:
    """
    创建并维护 LangGraph PostgreSQL Checkpointer。

    用途：
        - Workflow State 持久化
        - HITL Interrupt / Resume
        - Workflow Fault Recovery
        - Checkpoint / Resume
        - 后续 Time Travel / Debugging

    生命周期：

        create_checkpointer()
              ↓
        PostgreSQL Connection
              ↓
        setup()
              ↓
        LangGraph
              ↓
        checkpoint persistence
              ↓
        close connection
    """

    if not database_url:
        raise ValueError(
            "database_url must not be empty."
        )

    async with AsyncPostgresSaver.from_conn_string(
        database_url,
    ) as checkpointer:

        # 第一次使用时创建 checkpoint 相关表，
        # 同时执行必要的数据库 migration。
        await checkpointer.setup()

        yield checkpointer