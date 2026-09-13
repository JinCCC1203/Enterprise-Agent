from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
)

text = "请测试长期记忆 embedding。"

vector = embeddings.embed_query(text)

print("实际 embedding 维度:", len(vector))

client = getattr(embeddings, "_client", None)

print("SentenceTransformer:", client)

if client is not None:
    try:
        print(
            "模型声明维度:",
            client.get_sentence_embedding_dimension(),
        )
    except Exception as exc:
        print(
            "无法读取模型维度:",
            exc,
        )