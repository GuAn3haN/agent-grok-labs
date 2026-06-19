import json

from src.llm_client import LLMClient
from src.toolbox import Toolbox


class AgentLoop:
    """Think → Act → Observe 循环"""

    def __init__(self, llm_client: LLMClient, toolbox: Toolbox, max_rounds: int = 5):
        self.llm = llm_client
        self.toolbox = toolbox
        self.max_rounds = max_rounds

    def run(self, user_query: str) -> str:
        messages = [{"role": "user", "content": user_query}]
        action_count = 0

        while True:
            # Think
            response = self.llm.chat(messages, self.toolbox.get_definitions())

            if response["type"] == "text":
                return response["content"]

            if response["type"] == "tool_call":
                # 测试2的核心判断：如果已经达到最大行动次数，禁止再行动
                if action_count >= self.max_rounds:
                    # 连兜底都不给，直接告诉模型“不能再调了”，但这里我们返回提示
                    return "抱歉，已达到最大行动次数，无法继续执行工具。"

                tool_name = response["tool_name"]
                arguments = response["arguments"]
                tool_call_id = response["tool_call_id"]

                result = self.toolbox.execute(tool_name, arguments)
                action_count += 1

                # 追加 assistant 和 tool 消息（和之前一样）
                assistant_msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments, ensure_ascii=False)
                        }
                    }]
                }
                messages.append(assistant_msg)

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result
                }
                messages.append(tool_msg)