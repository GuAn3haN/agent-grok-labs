from collections import deque
from micromcp.server import MCPServer
from micromcp.client import MCPClient
from agent.llm_agent import LLMAgent

# 创建服务器，注册一个 weather 工具
server = MCPServer("WeatherServer")
def get_weather(city: str):
    return {"city": city, "temp": 28, "condition": "晴"}
server.register_tool("get_weather", "获取指定城市的天气",
                     {
                         "type": "object",
                         "properties": {"city": {"type": "string", "description": "城市名称"}},
                         "required": ["city"]
                     },
                     get_weather)

# 传输模拟
out = deque()
def write(msg: str):
    resp = server.process_message(msg)
    out.append(resp)
def read() -> str:
    return out.popleft()

client = MCPClient(read, write)
agent = LLMAgent(client)

# 测试对话
answer = agent.chat("北京天气怎么样？")
print("LLM 回答:", answer)