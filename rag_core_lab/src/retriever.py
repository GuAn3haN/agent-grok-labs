import numpy as np

from openai import OpenAI # type: ignore
import os

class BaseRetriever:
    def __init__(self, embedding_model: str = "text-embedding-v4"):
        self.embedding_model = embedding_model
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.chunks = []
        self.embeddings = []

    def index(self, chunks):
        self.chunks = chunks
        self.embeddings = []
        for chunk in chunks:
            resp = self.client.embeddings.create(
                model=self.embedding_model,   # "text-embedding-v4"
                input=chunk
            )
            emb = resp.data[0].embedding  # 返回的向量列表取第一个
            self.embeddings.append(np.array(emb))

    def retrieve(self, query, k=3):
        # 1. 将问题向量化
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        query_emb = np.array(resp.data[0].embedding)
        
        # 2. 计算余弦相似度
        similarities = []
        for emb in self.embeddings:
            dot = np.dot(query_emb, emb)
            norm = np.linalg.norm(query_emb) * np.linalg.norm(emb)
            sim = dot / norm if norm != 0 else 0.0
            similarities.append(sim)
        
        # 3. 按相似度降序取前k个的下标
        top_k_idx = np.argsort(similarities)[-k:][::-1]  # 倒序取最大
        return [self.chunks[i] for i in top_k_idx]