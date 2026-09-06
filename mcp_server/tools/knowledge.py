from mcp.server import MCPServer

def register_knowledge_tools(mcp: MCPServer) -> None:
    """注册知识库相关Tools"""

    @mcp.tool()
    def search_knowledge_base(query: str, k: int) -> list[dict]:
        """
        搜索企业知识库。

        Args:
            query: 用户的问题。
            top_k: 返回的最大结果数量。

        Returns:
            知识库检索结果。
        """
        documents = [
            {
                "title": "员工远程办公制度",
                "content": "公司支持符合条件的员工申请远程办公，具体天数由所属部门规定。",
                "source": "remote_work_policy.md",
            },
            {
                "title": "员工报销管理制度",
                "content": "员工出差和业务活动产生的费用，需要按照公司财务制度提交报销申请。",
                "source": "expense_policy.md",
            },
            {
                "title": "研发代码规范",
                "content": "研发人员提交代码前需要进行自测，重要功能需要经过 Code Review。",
                "source": "engineering_guidelines.md",
            },
        ]

        if not query.strip():
            return []

        results = []

        for document in documents:
            text = f"{document['title']} {document['content']}"

            if query.lower() in text.lower():
                results.append(document)

        return results[:k]
