import json
import os
from openai import OpenAI
from micromcp.client import MCPClient
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的环境变量

class LLMAgent:
    def __init__(self, mcp_client: MCPClient, model: str = "qwen-plus"):
        """
        初始化 LLM Agent，完成 MCP 握手并获取工具 schema。
        
        Args:
            mcp_client: 已配置好传输的 MCPClient 实例。
            model: 千问模型名称，可选 qwen-plus, qwen-max 等。
        """
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.mcp = mcp_client
        self.model = model

        # 握手并获取工具 schema（只需一次）
        self.mcp.initialize()
        self.tools_schema = self.mcp.tools_as_openai_schema()

    def chat(self, user_message: str, max_turns: int = 5) -> str:
        """
        与 LLM 对话，LLM 可决定调用 MCP 工具。实现简化的 Agent 循环：

        1. 初始化 messages = [{"role": "user", "content": user_message}]
        2. 循环最多 max_turns 次：
           a. 调用 OpenAI API，传入 tools_schema
           b. 若响应中有 tool_calls：
              - 遍历每个 tool_call，提取函数名和参数
              - 通过 self.mcp.call_tool 执行
              - 将结果作为 tool 消息追加到 messages
           c. 若无 tool_calls，返回 assistant 的文本回复
        3. 若达到最大轮次仍未得到最终文本，返回最后一条 assistant 消息内容。
        """
        messages = [{"role": "user", "content": user_message}]

        for _ in range(max_turns):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools_schema
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                # 将包含 tool_calls 的 assistant 消息加入历史
                messages.append(msg.model_dump())
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    # 通过 MCP 客户端执行工具调用
                    result = self.mcp.call_tool(func_name, **func_args)
                    
                    # 将工具结果包装为 tool 角色消息
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
            else:
                # 最终文本回复
                return msg.content if msg.content else ""

        # 达到最大轮次仍未得到文本，返回最后一条助手消息的内容
        last_msg = messages[-1]
        if last_msg.get("role") == "assistant" and "content" in last_msg:
            return last_msg["content"]
        return "达到最大轮次，未得到最终回复。"