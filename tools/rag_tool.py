from langchain.tools import tool
import requests

RAG_SERVICE_URL="http://localhost:8000/search"

@tool
def rag_search(query: str, k: int) -> str:
    """
    查询企业知识库

    当用户的问题涉及企业内部文档，产品信息，公司制度，
    企业知识库等内容时，使用该工具检索相关资料。

    Args
    :param query:用户需要查询的问题
    :param k:返回的相关文档数量
    """

    response=requests.post(
        RAG_SERVICE_URL,
        params={
            "query":query,
            "k":k
        }
    )

    response.raise_for_status()
    data=response.json()

    documents=data["documents"]

    context="\n\n".join(
        f"来源：{doc.metadata.get('source','未知')}"
        f"{doc.page_content}"
        for doc in documents
    )

    return context


