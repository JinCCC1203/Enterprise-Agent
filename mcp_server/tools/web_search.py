from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv
from mcp.server import MCPServer


load_dotenv()


TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def register_web_search_tools(mcp: MCPServer) -> None:
    """
    注册 Web Search MCP Tool。
    """

    @mcp.tool()
    def web_search(
        query: str,
        max_results: int = 5,
    ) -> dict[str, Any]:
        """
        搜索互联网公开信息。

        该工具用于：
        - 当前外部信息查询
        - 公开技术资料搜索
        - 官方文档检索
        - 新闻与实时信息研究
        - 外部知识汇总

        Args:
            query:
                搜索查询。

            max_results:
                返回的最大结果数量，范围 1-10。

        Returns:
            包含搜索结果、标题、URL 和摘要的结构化结果。
        """

        query = query.strip()

        if not query:
            return {
                "success": False,
                "error": "query cannot be empty",
            }

        max_results = max(
            1,
            min(max_results, 10),
        )

        api_key = os.getenv(
            "TAVILY_API_KEY"
        )

        if not api_key:
            return {
                "success": False,
                "error": (
                    "TAVILY_API_KEY is not configured."
                ),
            }

        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "topic": "general",
            "include_answer": True,
            "include_raw_content": False,
            "include_images": False,
        }

        try:
            response = requests.post(
                TAVILY_SEARCH_URL,
                json=payload,
                timeout=20,
            )

            response.raise_for_status()

        except requests.RequestException as exc:
            return {
                "success": False,
                "error": (
                    f"web search request failed: {exc}"
                ),
            }

        try:
            data = response.json()

        except ValueError:
            return {
                "success": False,
                "error": (
                    "web search returned invalid JSON."
                ),
            }

        results = []

        for item in data.get(
            "results",
            [],
        ):

            results.append(
                {
                    "title": item.get(
                        "title",
                        "",
                    ),
                    "url": item.get(
                        "url",
                        "",
                    ),
                    "content": item.get(
                        "content",
                        "",
                    ),
                    "score": item.get(
                        "score",
                    ),
                }
            )

        return {
            "success": True,
            "query": query,
            "answer": data.get(
                "answer",
            ),
            "results": results,
        }