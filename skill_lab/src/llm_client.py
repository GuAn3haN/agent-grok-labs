from openai import OpenAI # type: ignore
from dotenv import load_dotenv # type: ignore
import os
import json


load_dotenv()  # 自动读取 .env 文件

class LLMClient:
    """封装模型调用，区分工具调用和文本回复"""

    def __init__(self, model: str = "qwen-plus"):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model

    def chat(self, messages: list[dict], tools: list[dict] = None) -> dict:
        """
        返回：
          - 工具调用: {"type": "tool_call", "tool_name": str, "arguments": dict, "tool_call_id": str}
          - 文本回复: {"type": "text", "content": str}
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        # 判断是否要调用工具
        if msg.tool_calls:
            # 只处理第一个工具调用（顺序执行）
            tool_call = msg.tool_calls[0]
            tool_name = tool_call.function.name
            # arguments 是 JSON 字符串，需要解析
            arguments = json.loads(tool_call.function.arguments)
            return {
                "type": "tool_call",
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_call_id": tool_call.id
            }
        else:
            return {
                "type": "text",
                "content": msg.content
            }