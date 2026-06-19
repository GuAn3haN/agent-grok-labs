from openai import OpenAI # type: ignore
import os

class RAGGenerator:
    def __init__(self, retriever, model: str = "qwen-plus"):
        self.retriever = retriever
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def answer(self, question, k=3, return_context=False):
        # 1. 检索
        chunks = self.retriever.retrieve(question, k=k)
        
        # 2. 组装 prompt
        context_str = "\n".join(f"- {chunk}" for chunk in chunks)
        prompt = f"""请根据以下参考资料回答问题。如果资料中没有信息，则明确告知不知道。

    参考资料：
    {context_str}

    问题：{question}

    答案："""
        
        # 3. 调用千问生成回答
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip()
        
        # 4. 返回
        if return_context:
            return {"answer": answer, "chunks": chunks}
        return answer